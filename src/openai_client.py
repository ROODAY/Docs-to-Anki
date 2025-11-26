import os
import time
import json
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
import multiprocessing

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt5-mini")
OPENAI_REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))

# optional comma-separated fallback models to try if primary model is not available
FALLBACK_MODELS = [m.strip() for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-4o-mini,gpt-4,gpt-3.5-turbo").split(",") if m.strip()]

# Models we should avoid trying for the remainder of this process run
# (populated when a model returns an unrecoverable error such as model_not_found
# or unsupported parameters for that model).
BLOCKED_MODELS = set()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill it.")

# For openai>=1.0, it's recommended to create a client, but setting api_key is still supported
openai.api_key = OPENAI_API_KEY


def make_prompt_for_section(section_text: str, lesson_date: str) -> List[Dict[str, str]]:
    system = (
        "You are an assistant that converts a lesson (Tamil learner, English speaker) into Anki flashcards. "
        "Return only a JSON array of objects, no surrounding explanation. Each object must have: 'front', 'back', and optional 'tags' (a string of comma-separated tags)."
    )

    user = (
        f"Lesson date: {lesson_date}\n\nHere is the lesson content in markdown:\n\n{section_text}\n\n"
            "Create a diverse set of cards useful for language learning: sentence -> translation, cloze deletions for key vocabulary, vocab (Tamil) -> meaning (English), example sentence recognition, conjugation/declension tips, and pronunciation hints. "
            "Keep each 'front' and 'back' concise. For 'tags' include 'tamil' and the lesson date. Return only a JSON array of objects with keys 'front' and 'back'."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _openai_child_runner(q, messages, use_max_completion: bool, include_temperature: bool, model_name: str):
    """Top-level child runner for multiprocessing (must be picklable on Windows)."""
    try:
        kwargs = {
            "model": model_name,
            "messages": messages,
        }
        if include_temperature:
            kwargs["temperature"] = 0.2
        if use_max_completion:
            kwargs["max_completion_tokens"] = 1200
        else:
            kwargs["max_tokens"] = 1200

        if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
            resp = openai.chat.completions.create(**kwargs)
        else:
            resp = openai.ChatCompletion.create(**kwargs)

        # try to extract content conservatively and send back the string
        try:
            # object-style
            choices = getattr(resp, "choices", None)
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content is None:
                        try:
                            content = msg["content"]
                        except Exception:
                            content = None
                    if content:
                        q.put(("ok", str(content)))
                        return
        except Exception:
            pass
        try:
            # dict-style
            content = resp["choices"][0]["message"]["content"]
            q.put(("ok", str(content)))
            return
        except Exception:
            pass

        # final fallback: return repr
        q.put(("err", f"unextractable_response:{repr(resp)[:200]}"))
    except Exception as e:
        q.put(("err", str(e)))


def call_openai_for_section(section_text: str, lesson_date: str, max_retries: int = 4) -> List[Dict[str, Any]]:
    messages = make_prompt_for_section(section_text, lesson_date)
    backoff = 1.0
    # choose the starting model: try primary unless it's been blocked for this run
    model_to_try = OPENAI_MODEL if OPENAI_MODEL not in BLOCKED_MODELS else None
    # prepare fallback list skipping any blocked models
    available_fallbacks = [m for m in FALLBACK_MODELS if m not in BLOCKED_MODELS]
    fallback_idx = 0
    last_err = None

    for attempt in range(max_retries):
        try:
            # We'll run the actual OpenAI network call in a child process so that
            # KeyboardInterrupt in the parent is respected immediately and we can
            # enforce a per-request timeout. The actual child runner is implemented
            # at module scope as `_openai_child_runner` to be picklable on Windows.

            def _call_with_param(use_max_completion: bool = False, include_temperature: bool = True):
                # ensure model to try is set
                nonlocal model_to_try
                if not model_to_try:
                    if available_fallbacks:
                        model_to_try = available_fallbacks[0]
                    else:
                        raise RuntimeError("No model available to try")

                q = multiprocessing.Queue()
                p = multiprocessing.Process(target=_openai_child_runner, args=(q, messages, use_max_completion, include_temperature, model_to_try))
                p.start()
                try:
                    p.join(OPENAI_REQUEST_TIMEOUT)
                    if p.is_alive():
                        p.terminate()
                        p.join()
                        raise RuntimeError(f"OpenAI call timed out after {OPENAI_REQUEST_TIMEOUT}s for model {model_to_try}")
                    # got result
                    try:
                        status, payload = q.get_nowait()
                    except Exception:
                        raise RuntimeError("No response returned from OpenAI child process")
                    if status == "ok":
                        # return a minimal dict-like object with choices/message/content to keep
                        # compatibility with existing extraction path
                        return {"choices": [{"message": {"content": payload}}]}
                    else:
                        raise RuntimeError(payload)
                except KeyboardInterrupt:
                    # If parent received SIGINT, make sure child is cleaned up and re-raise
                    try:
                        if p.is_alive():
                            p.terminate()
                            p.join()
                    except Exception:
                        pass
                    raise

            # Try a sequence of parameter variants to accommodate different model requirements:
            # 1) max_tokens + temperature
            # 2) max_completion_tokens + temperature
            # 3) max_tokens without temperature
            # 4) max_completion_tokens without temperature
            try:
                resp = _call_with_param(use_max_completion=False, include_temperature=True)
            except Exception as call_err:
                msg = str(call_err)
                # If the model returns errors that indicate it's unusable for our parameters
                # (not accessible, unsupported params, verification required), block it for
                # subsequent calls and move to the next model.
                unrecoverable = False
                if any(k in msg for k in ("model_not_found", "Model not found", "Your organization must be verified")):
                    unrecoverable = True
                if any(k in msg for k in ("Unsupported parameter", "max_tokens", "Unsupported value", "temperature")):
                    # these might be recoverable by changing params; we'll attempt retries below
                    unrecoverable = False

                if unrecoverable:
                    BLOCKED_MODELS.add(model_to_try)
                    # pick next fallback model if available
                    if fallback_idx < len(available_fallbacks):
                        model_to_try = available_fallbacks[fallback_idx]
                        fallback_idx += 1
                        # try next model in the outer loop
                        continue
                    else:
                        # no more models to try
                        raise

                # Otherwise attempt parameter variations as before
                if "max_tokens" in msg or "Unsupported parameter" in msg:
                    try:
                        resp = _call_with_param(use_max_completion=True, include_temperature=True)
                    except Exception as call_err2:
                        msg2 = str(call_err2)
                        if "temperature" in msg2 or "Unsupported value" in msg2:
                            resp = _call_with_param(use_max_completion=True, include_temperature=False)
                        else:
                            # if this also fails, block this model and move on
                            BLOCKED_MODELS.add(model_to_try)
                            if fallback_idx < len(available_fallbacks):
                                model_to_try = available_fallbacks[fallback_idx]
                                fallback_idx += 1
                                continue
                            else:
                                raise
                elif "temperature" in msg or "Unsupported value" in msg:
                    try:
                        resp = _call_with_param(use_max_completion=False, include_temperature=False)
                    except Exception:
                        BLOCKED_MODELS.add(model_to_try)
                        if fallback_idx < len(available_fallbacks):
                            model_to_try = available_fallbacks[fallback_idx]
                            fallback_idx += 1
                            continue
                        else:
                            raise
                else:
                    # unknown error — block this model to avoid repeated failures
                    BLOCKED_MODELS.add(model_to_try)
                    if fallback_idx < len(available_fallbacks):
                        model_to_try = available_fallbacks[fallback_idx]
                        fallback_idx += 1
                        continue
                    else:
                        raise

            # extract content in a way that supports both the new object-style
            # responses and the older dict-like responses
            def _get_response_content(rsp):
                # try attribute access first
                try:
                    choices = getattr(rsp, "choices", None)
                    if choices:
                        first = choices[0]
                        msg = getattr(first, "message", None)
                        if msg is not None:
                            # message may have attribute 'content' or be dict-like
                            content = getattr(msg, "content", None)
                            if content is None:
                                try:
                                    content = msg["content"]
                                except Exception:
                                    content = None
                            if content:
                                return content.strip()
                except Exception:
                    pass
                # try dict-like access
                try:
                    return rsp["choices"][0]["message"]["content"].strip()
                except Exception:
                    pass
                raise RuntimeError("Unable to extract content from OpenAI response")

            content = _get_response_content(resp)

            # If the model returned no content (empty string), raise a clearer error
            if not content or not str(content).strip():
                # Try to capture a small preview of the raw response for debugging
                raw_preview = None
                try:
                    raw_preview = getattr(resp, "text", None)
                except Exception:
                    raw_preview = None
                if not raw_preview:
                    try:
                        raw_preview = repr(resp)
                    except Exception:
                        raw_preview = "<unprintable response>"
                raise RuntimeError(
                    f"Empty content returned by model {model_to_try}. Raw response preview: {raw_preview[:100]!r}"
                )

            # Expect JSON array
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed
                else:
                    # not a list, try to extract JSON substring
                    start = content.find("[")
                    end = content.rfind("]")
                    if start != -1 and end != -1:
                        parsed2 = json.loads(content[start:end+1])
                        return parsed2
            except Exception:
                # fallback: try to be forgiving by returning an error
                raise
        except Exception as e:
            # If the error indicates the model doesn't exist, try fallbacks (if any)
            last_err = e
            try:
                err_code = getattr(e, 'code', None)
            except Exception:
                err_code = None
            # openai.NotFoundError or model_not_found
            if (err_code == 'model_not_found') or e.__class__.__name__ in ('NotFoundError',):
                if fallback_idx < len(FALLBACK_MODELS):
                    model_to_try = FALLBACK_MODELS[fallback_idx]
                    fallback_idx += 1
                    # don't count this as one of the exponential backoff retries; try immediately
                    continue
            wait = backoff * (2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"OpenAI request failed after retries: {last_err}")
