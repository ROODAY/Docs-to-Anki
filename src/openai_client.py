import os
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# default to exact model name requested by user
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))

# simple logger for visibility
logger = logging.getLogger("docs_to_anki.openai_client")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill it.")

openai.api_key = OPENAI_API_KEY


def make_prompt_for_section(section_text: str, lesson_date: str) -> List[Dict[str, str]]:
    system = (
        "You are an assistant that converts a lesson (Tamil learner, English speaker) into Anki flashcards."
    )
    user = (
        f"Lesson date: {lesson_date}\n\n{section_text}\n\n"
        "Return ONLY a JSON array of objects with keys 'front' and 'back' and optional 'tags'."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _call_openai_async(messages, model_name: str):
    # Async call using AsyncOpenAI so KeyboardInterrupt can cancel the
    # running coroutine. Use generous token limit to avoid truncation.
    client = AsyncOpenAI()
    try:
        coro = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=4000,
        )
        resp = await asyncio.wait_for(coro, timeout=OPENAI_REQUEST_TIMEOUT)
        return resp
    finally:
        # close the client if it exposes aclose
        try:
            await client.aclose()
        except Exception:
            pass


def _spawn_call(messages, model_name, timeout: int):
    # Backwards-compatible sync wrapper: run the async call in the event loop.
    return asyncio.run(_call_openai_async(messages, model_name))


async def call_openai_for_section(section_text: str, lesson_date: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    """Call the configured model (no fallbacks). Retries only vary request parameters.

    This enforces using only `OPENAI_MODEL` (e.g. `gpt5-mini`) as requested.
    """
    messages = make_prompt_for_section(section_text, lesson_date)
    model = OPENAI_MODEL
    last_err = None

    logger.info("Calling OpenAI model=%s for lesson=%s (max_retries=%d)", model, lesson_date, max_retries)

    @retry(stop=stop_after_attempt(max_retries), wait=wait_exponential(multiplier=1, min=1, max=30),
           retry=retry_if_exception_type(Exception), reraise=True)
    async def _attempt_call():
        logger.info("Calling async OpenAI: max_completion_tokens (no temperature) (timeout=%ss)", OPENAI_REQUEST_TIMEOUT)
        return await _call_openai_async(messages, model)

    try:
        resp = await _attempt_call()
    except Exception as e:
        raise RuntimeError(f"OpenAI request failed after retries: {e}")

    # extract content
    try:
        # attempt dict-style extraction first
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            # try object-style
            choices = getattr(resp, "choices", None)
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                else:
                    content = None
            else:
                content = None
    except Exception:
        # log raw response for debugging then raise
        try:
            raw = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
        except Exception:
            raw = repr(resp)
        logger.error("Unable to extract content from OpenAI response. Raw response: %s", raw)
        raise RuntimeError("Unable to extract content from OpenAI response")

    if content is None or not str(content).strip():
        try:
            raw = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
        except Exception:
            raw = repr(resp)
        logger.error("Empty content returned by model. Raw response: %s", raw)
        raise RuntimeError("Empty content returned by model")

    # defensive cleaning: strip BOM and fences
    content = str(content)
    content = content.lstrip('\ufeff').strip()
    if content.startswith("```"):
        parts = content.splitlines()
        # drop first fence line
        parts = parts[1:]
        if parts and parts[-1].strip().startswith('```'):
            parts = parts[:-1]
        content = '\n'.join(parts).strip()

    logger.info("Received content length=%d", len(content) if content else 0)
    # parse JSON array
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            logger.info("Parsed JSON array with %d items", len(parsed))
            return parsed
    except Exception:
        logger.info("JSON parse failed, attempting bracketed-substring extraction")
        s = content.find("[")
        e = content.rfind("]")
        if s != -1 and e != -1 and e > s:
            parsed = json.loads(content[s:e+1])
            if isinstance(parsed, list):
                logger.info("Parsed JSON from substring with %d items", len(parsed))
                return parsed
        try:
            raw = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
        except Exception:
            raw = repr(resp)
        logger.error("Unable to parse JSON response from model. Raw response: %s", raw)
        raise RuntimeError("Unable to parse JSON response from model")
