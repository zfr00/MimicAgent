import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from fastapi import FastAPI, HTTPException
from tools.process_runner import backend_headers, remaining_seconds

from tools.request_models import QueryRequest, SearchRequest

FATHOM_BASE_URL = os.environ.get("FATHOM_BASE_URL", "http://127.0.0.1:8904")
WEB_TIMEOUT = float(os.environ.get("WEB_SEARCH_TIMEOUT", "350"))
WEB_MAX_TOKENS = int(os.environ.get("WEB_MAX_TOKENS", "3000"))
app = FastAPI(title="web_search (Fathom-DeepResearch)")


@lru_cache(maxsize=1)
def tokenizer():
    from pathlib import Path
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "cache/tokenizers"))
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def token_count(value):
    return len(tokenizer().encode(json.dumps(value, ensure_ascii=False), disallowed_special=()))


def bounded_result(value, text_key=None):
    """Cap the entire JSON payload in tokens while preserving valid JSON and URLs."""
    if WEB_MAX_TOKENS < 128:
        raise ValueError("WEB_MAX_TOKENS must be at least 128")
    if text_key:
        ids = tokenizer().encode(value[text_key], disallowed_special=())
        while token_count(value) > WEB_MAX_TOKENS and ids:
            ids = ids[:max(0, len(ids) - max(1, token_count(value) - WEB_MAX_TOKENS))]
            value[text_key] = tokenizer().decode(ids)
    else:
        while value['hits'] and token_count(value) > WEB_MAX_TOKENS:
            value['hits'].pop()
    if token_count(value) > WEB_MAX_TOKENS:
        raise HTTPException(502, "Web response metadata exceeds token limit")
    return value


def fathom_call(function, arguments):
    if function not in {"search_urls", "query_url"}:
        raise ValueError("Unsupported Fathom function")
    call = function + '(' + ', '.join(f'{key}={value!r}' for key, value in arguments.items()) + ')'
    try:
        response = requests.post(FATHOM_BASE_URL.rstrip('/') + '/execute',
            json={"env": f"from search_api import {function}", "call": call, "timeout": remaining_seconds(WEB_TIMEOUT)},
            headers=backend_headers(), timeout=remaining_seconds(WEB_TIMEOUT) + 1)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(502, f"Fathom request failed: {type(exc).__name__}") from exc
    result = data.get('result')
    if data.get('error') or not isinstance(result, str):
        raise HTTPException(502, "Fathom tool failed")
    if result.lower().startswith(('[tool-error]', '[error', '[failed', '[unable')):
        raise HTTPException(502, result)
    return result


@app.get('/health')
def health():
    try:
        response = requests.get(FATHOM_BASE_URL.rstrip('/') + '/openapi.json', timeout=5)
        response.raise_for_status()
        ready = '/execute' in response.json().get('paths', {})
    except (requests.RequestException, ValueError):
        ready = False
    return {'status': 'ok' if ready else 'unavailable', 'backend': 'Fathom-DeepResearch',
            'base_url': FATHOM_BASE_URL, 'max_tokens': WEB_MAX_TOKENS}


@app.post('/search')
def search(req: SearchRequest):
    text = fathom_call('search_urls', {'query': req.query, 'top_k': req.top_k})
    pattern = r'^### \d+\. (.*?)\n\*\*URL\*\*:\s*(\S+)\s*\n\n\*\*Snippet\*\*:\s*(.*?)(?=\n---\n|\Z)'
    hits = [{'title': title.strip(), 'url': url, 'snippet': snippet.strip()}
            for title, url, snippet in re.findall(pattern, text, re.M | re.S)]
    if text.strip() and not hits:
        raise HTTPException(502, 'Unexpected Fathom search result format')
    return bounded_result({'hits': hits[:req.top_k]})


@app.post('/query')
def query(req: QueryRequest):
    result = fathom_call('query_url', {'url': req.url, 'goal': req.goal})
    return bounded_result({'extracted_text': result, 'source': req.url}, 'extracted_text')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8002)
