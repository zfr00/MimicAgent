"""Cached Chinese-to-English exemplar text translation."""

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time

from openai import APIError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
TEXT_FIELDS = ("question", "answer", "analysis", "knowledge")
VERSION = "exemplar-en-v2"
PROMPT = """Translate the supplied Chinese scientific exercises into English.
Return one JSON object with a 'translations' array, in exactly the input order.
Each element must have exactly question, answer, analysis, knowledge string fields.
Translate all Chinese text into English, including natural-language text inside
LaTeX formulas, subscripts, superscripts, and reaction conditions. Preserve the
scientific meaning, mathematical structure, LaTeX commands, numbers, units,
symbols, and line breaks. Keep every <image> placeholder in its original position
relative to the surrounding meaning. Preserve empty strings."""

LOG = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    pass


def _remaining(default):
    deadline = os.environ.get("ARIS_TOOL_DEADLINE_UNIX")
    remaining = min(default, float(deadline) - time.time() - 2) if deadline else default
    if remaining <= 0:
        raise TranslationError("Translation time budget exhausted")
    return remaining


def _validate_translations(reply, source):
    if not isinstance(reply, dict) or set(reply) != {"translations"}:
        raise ValueError("Translation must return a translations object")
    rows = reply["translations"]
    if not isinstance(rows, list) or len(rows) != len(source):
        raise ValueError("Translation changed the number of exemplars")
    translations = []
    for original, row in zip(source, rows, strict=True):
        if not isinstance(row, dict) or set(row) != set(TEXT_FIELDS):
            raise ValueError("Translation changed its text fields")
        translated = {}
        for field in TEXT_FIELDS:
            value = row[field]
            if not isinstance(value, str) or bool(value.strip()) != bool(original[field].strip()):
                raise ValueError(f"Invalid translated {field}")
            if re.search(r"[\u3400-\u9fff]", value):
                raise ValueError(f"Translator left Chinese text in {field}")
            if value.count("<image>") != original[field].count("<image>"):
                raise ValueError(f"Translation changed image placeholders in {field}")
            translated[field] = value
        translations.append(translated)
    return translations


def _settings():
    settings = {
        "model": os.environ.get("ARIS_TRANSLATOR_MODEL", "gpt-4.1-mini"),
        "base_url": os.environ.get("ARIS_TRANSLATOR_BASE_URL", os.environ.get("ARIS_PRIMARY_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))),
        "api_key": os.environ.get("ARIS_TRANSLATOR_API_KEY", os.environ.get("ARIS_PRIMARY_API_KEY", os.environ.get("OPENAI_API_KEY", ""))),
        "token_parameter": os.environ.get("ARIS_TRANSLATOR_TOKEN_PARAMETER", "max_tokens"),
        "extra_body": json.loads(os.environ.get("ARIS_TRANSLATOR_EXTRA_BODY", "{}")),
        "attempts": int(os.environ.get("EXEMPLAR_TRANSLATION_RETRIES", "3")),
        "timeout": float(os.environ.get("EXEMPLAR_TRANSLATION_TIMEOUT", "60")),
        "max_tokens": int(os.environ.get("EXEMPLAR_TRANSLATION_MAX_TOKENS", "8192")),
    }
    if settings["token_parameter"] not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("Unsupported translator TOKEN_PARAMETER")
    if min(settings["attempts"], settings["timeout"], settings["max_tokens"]) <= 0:
        raise ValueError("Translation retry, timeout and token limits must be positive")
    if not isinstance(settings["extra_body"], dict):
        raise ValueError("Translator EXTRA_BODY must be an object")
    return settings


def _request(payload, settings):
    if not settings["api_key"]:
        raise TranslationError("Set ARIS_TRANSLATOR_API_KEY, ARIS_PRIMARY_API_KEY or OPENAI_API_KEY")
    last_error = None
    with OpenAI(api_key=settings["api_key"], base_url=settings["base_url"],
                timeout=settings["timeout"], max_retries=0) as client:
        for attempt in range(settings["attempts"]):
            try:
                response = client.chat.completions.create(
                    model=settings["model"], temperature=0,
                    messages=[{"role": "system", "content": PROMPT},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    response_format={"type": "json_object"}, extra_body=settings["extra_body"],
                    timeout=_remaining(settings["timeout"]),
                    **{settings["token_parameter"]: settings["max_tokens"]},
                )
                if not response.choices or response.choices[0].finish_reason != "stop":
                    raise ValueError("Incomplete translation JSON response")
                content = response.choices[0].message.content
                translated = json.loads(content or "")
                _validate_translations(translated, payload)
                return translated
            except (APIError, ValueError, TypeError) as exc:
                last_error = exc
                if attempt + 1 < settings["attempts"]:
                    time.sleep(_remaining(min(.5 * 2 ** attempt, 4)))
    raise TranslationError(f"Exemplar translation failed after {settings['attempts']} attempts ({type(last_error).__name__})") from last_error


@contextmanager
def _cache_lock(path):
    stream = None
    try:
        stream = path.open("a")
        end = time.monotonic() + _remaining(360)
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= end:
                    raise TranslationError("Translation cache lock timed out")
                time.sleep(min(.05, _remaining(.05)))
    except OSError as exc:
        if stream is not None:
            stream.close()
            stream = None
        LOG.warning("Translation cache lock unavailable (%s)", type(exc).__name__)
    except BaseException:
        if stream is not None:
            stream.close()
        raise
    try:
        yield
    finally:
        if stream is not None:
            stream.close()


def _save_cache(destination, cached):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.stem, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(cached, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError as exc:
        LOG.warning("Translation cache write failed (%s)", type(exc).__name__)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                LOG.warning("Translation temporary file cleanup failed (%s)", type(exc).__name__)


def translate_exemplars(exemplars):
    """Translate selected exemplars once per source/config revision, across workers."""
    try:
        settings = _settings()
        cache_dir = Path(os.environ.get("EXEMPLAR_TRANSLATION_CACHE_DIR", ROOT / "cache/exemplar_translations")).resolve()
        if not cache_dir.is_relative_to(ROOT):
            raise ValueError("EXEMPLAR_TRANSLATION_CACHE_DIR must be inside the code repository")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOG.warning("Translation cache directory unavailable (%s)", type(exc).__name__)
        payload = [{field: exemplar[field] for field in TEXT_FIELDS} for exemplar in exemplars]
        identity = {"version": VERSION, "source": exemplars,
                    **{key: settings[key] for key in ("model", "base_url", "token_parameter", "extra_body")}}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        destination = cache_dir / f"{digest}.json"
        with _cache_lock(cache_dir / f"{digest}.lock"):
            translated = None
            try:
                cached = json.loads(destination.read_text(encoding="utf-8"))
                translated = _validate_translations(cached, payload)
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError, KeyError) as exc:
                LOG.warning("Translation cache read failed (%s)", type(exc).__name__)
            if translated is None:
                cached = _request(payload, settings)
                translated = _validate_translations(cached, payload)
                _save_cache(destination, cached)
        result = copy.deepcopy(exemplars)
        for exemplar, fields in zip(result, translated, strict=True):
            exemplar.update(fields)
        return result
    except TranslationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise TranslationError(f"Exemplar translation failed ({type(exc).__name__})") from exc
