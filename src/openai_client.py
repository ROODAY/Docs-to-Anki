import os
import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI, APIError, RateLimitError, InternalServerError
from dotenv import load_dotenv
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import tiktoken
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))
OUTPUT_DIR = os.path.abspath(os.getenv("OUTPUT_DIR", "outputs"))

logger = logging.getLogger("docs_to_anki.openai_client")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def make_prompt_for_section(section_text: str, lesson_date: str):
    """Constructs messages for the OpenAI chat.completions API."""
    system = (
        "You convert Tamil lesson notes into strict JSON flashcards for language learning. "
        "ONLY output JSON. No extra text. "
        "Format: an array of objects with keys: 'front', 'back', optional 'tags'. "
        "\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Language separation: Each card must have one side ONLY in Tamil and the other side ONLY in English. "
        "   This enables reverse card format. Do not mix languages on the same side.\n"
        "2. Use standard ASCII punctuation: Use only standard ASCII characters for punctuation. "
        "   Avoid Unicode quote characters (curly quotes like ' ' \" \") and other fancy Unicode characters "
        "   that may display as HTML entity codes in Anki. Standard ASCII quotes (\"), apostrophes ('), "
        "   and other ASCII punctuation are fine and will display normally.\n"
        "3. One example per card: If there are multiple examples or phrases, create separate cards for each one. "
        "   Do not combine multiple examples into a single card.\n"
        "4. Filter content: Only create cards from content that makes sense as language learning material. "
        "   Skip meta-text like 'next lesson we will learn', 'in this chapter', 'remember that', "
        "   or other instructional text that is not actual language content (vocabulary, phrases, sentences)."
    )
    user = (
        f"Lesson date: {lesson_date}\n\n"
        f"{section_text}\n\n"
        "Create flashcards following all requirements. Output only valid JSON array. "
        "Do not wrap in markdown. Do not add explanations."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def _call_openai_async(
    messages, model_name: str, max_completion_tokens: int = 4000
):
    """Makes the actual OpenAI request with timeout handling."""
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
            ),
            timeout=OPENAI_REQUEST_TIMEOUT,
        )
        return resp

    except asyncio.TimeoutError:
        logger.error("OpenAI call timed out after %s seconds", OPENAI_REQUEST_TIMEOUT)
        raise
    except asyncio.CancelledError:
        logger.info("Request cancelled")
        raise
    except Exception:
        logger.exception("Unexpected OpenAI API error")
        raise


async def call_openai_for_section(
    section_text: str,
    lesson_date: str,
    max_retries: int = 3,
    max_completion_tokens: int = 4000,
) -> List[Dict[str, Any]]:
    messages = make_prompt_for_section(section_text, lesson_date)
    model = OPENAI_MODEL

    # compute request token count once (before firing the request)
    try:
        encoding = tiktoken.encoding_for_model(model)
        joined = "\n".join(
            [f"{m.get('role', '')}: {m.get('content', '')}" for m in messages]
        )
        request_token_count = len(encoding.encode(joined))
    except Exception:
        request_token_count = None

    logger.info(
        "Calling OpenAI model=%s for lesson=%s (max_retries=%d, timeout=%ss, max_completion_tokens=%d)",
        model,
        lesson_date,
        max_retries,
        OPENAI_REQUEST_TIMEOUT,
        max_completion_tokens,
    )
    if request_token_count is not None:
        logger.info("Request token count=%d for model=%s", request_token_count, model)
    else:
        logger.info("Request token count=unknown for model=%s", model)

    attempts = {"count": 0}

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(
            (asyncio.TimeoutError, APIError, RateLimitError, InternalServerError)
        ),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _attempt_call():
        attempts["count"] += 1
        logger.info("Attempt %d/%d", attempts["count"], max_retries)
        return await _call_openai_async(
            messages, model, max_completion_tokens=max_completion_tokens
        )

    try:
        resp = await _attempt_call()
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed after retries: {e}")

    # If the model explicitly signalled truncation (finish_reason == 'length'),
    # record the failure and move on without retrying.
    try:
        # Robustly extract the first choice and its finish_reason regardless of
        # whether the response is a mapping or an object with attributes.
        def _get_choice0_and_finish(r):
            try:
                if hasattr(r, "choices"):
                    choices = r.choices
                elif isinstance(r, dict):
                    choices = r.get("choices")
                else:
                    choices = getattr(r, "to_dict", lambda: dict(r))().get("choices")
                if choices and len(choices) > 0:
                    choice0 = choices[0]
                else:
                    choice0 = None
            except Exception:
                choice0 = None
            finish = None
            try:
                if isinstance(choice0, dict):
                    finish = choice0.get("finish_reason")
                else:
                    finish = getattr(choice0, "finish_reason", None)
            except Exception:
                finish = None
            return choice0, finish

        choice0, finish_reason = _get_choice0_and_finish(resp)
        if finish_reason == "length":
            token_count = request_token_count
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fail_path = os.path.join(OUTPUT_DIR, "failures.log")
            ts = datetime.utcnow().isoformat()
            try:
                with open(fail_path, "a", encoding="utf-8") as ff:
                    ff.write(
                        f"{ts} | {lesson_date} | tokens={token_count or 'unknown'} | finish_reason=length\n"
                    )
            except Exception:
                logger.warning("Failed to write failure log to %s", fail_path)

            logger.error(
                "Model response truncated (finish_reason=length) for lesson=%s; logged to %s",
                lesson_date,
                fail_path,
            )
            raise RuntimeError("Model response truncated (finish_reason=length)")
    except Exception:
        # If anything goes wrong during finish_reason inspection, continue
        # to normal extraction logic so we surface the original error.
        pass

    # extract content safely
    try:
        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Empty content returned by model")
    except Exception:
        raw = getattr(resp, "to_dict", lambda: dict(resp))()
        logger.error("Unable to extract content from response: %s", raw)

        # Write a failure entry to failures.log and also persist the full raw
        # response for debugging.
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fail_path = os.path.join(OUTPUT_DIR, "failures.log")
            ts = datetime.utcnow().isoformat()
            token_count = request_token_count
            with open(fail_path, "a", encoding="utf-8") as ff:
                ff.write(
                    f"{ts} | {lesson_date} | tokens={token_count or 'unknown'} | reason=extraction_failed\n"
                )
            # also write full raw response to a separate file for inspection
            safe_date = (lesson_date or "unknown").replace("/", "-")
            raw_path = os.path.join(
                OUTPUT_DIR, f"failure-{safe_date}-{ts.replace(':', '-')}.json"
            )
            try:
                with open(raw_path, "w", encoding="utf-8") as rf:
                    json.dump(raw, rf, ensure_ascii=False, indent=2)
            except Exception:
                logger.warning("Failed to write raw failure response to %s", raw_path)
        except Exception:
            logger.warning("Failed to record failure to %s", OUTPUT_DIR)

        raise RuntimeError("Unable to extract content from OpenAI response")

    # clean content
    content = str(content).lstrip("\ufeff").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:] if len(lines) > 1 else []
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    logger.info("Received content length=%d", len(content))

    # parse JSON
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError("JSON is not a list")
        logger.info("Parsed JSON array with %d items", len(parsed))
        return parsed
    except Exception:
        # fallback: extract first bracketed JSON substring
        s = content.find("[")
        e = content.rfind("]")
        if s != -1 and e != -1 and e > s:
            parsed = json.loads(content[s : e + 1])
            if isinstance(parsed, list):
                logger.info("Parsed JSON from substring with %d items", len(parsed))
                return parsed
        raw = getattr(resp, "to_dict", lambda: dict(resp))()
        logger.error("Failed to parse JSON. Raw response: %s", raw)
        raise RuntimeError("Model did not return valid JSON in parsed field")
