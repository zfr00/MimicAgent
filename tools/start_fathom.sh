#!/usr/bin/env bash
set -euo pipefail
FATHOM_TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FATHOM_ROOT="${FATHOM_TOOL_DIR}/vendor/fathom"
: "${SERPER_API_KEY:?Set SERPER_API_KEY}"
export OPENAI_API_KEY="${WEB_LLM_API_KEY:-${ARIS_PRIMARY_API_KEY:-${OPENAI_API_KEY:-}}}"
: "${OPENAI_API_KEY:?Set WEB_LLM_API_KEY or ARIS_PRIMARY_API_KEY}"
export OPENAI_BASE_URL="${WEB_LLM_BASE_URL:-${ARIS_PRIMARY_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}}"
export QUERY_LLM="openai:${WEB_LLM_MODEL:-gpt-4.1-mini}"
export SERPER_CACHE_DIR="${FATHOM_TOOL_DIR}/../cache/web_search/serper"
export JINA_CACHE_DIR="${FATHOM_TOOL_DIR}/../cache/web_search/jina"
export TIKTOKEN_CACHE_DIR="${FATHOM_TOOL_DIR}/../cache/tokenizers"
export CRAWL4_AI_BASE_DIRECTORY="${FATHOM_TOOL_DIR}/../cache/web_search"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-${FATHOM_TOOL_DIR}/../cache/browsers}"
export PYTHONPATH="${FATHOM_TOOL_DIR}/..:${FATHOM_ROOT}/serving/web_agents:${FATHOM_ROOT}/serving${PYTHONPATH:+:${PYTHONPATH}}"
export MAX_OUTBOUND="${MAX_OUTBOUND:-16}"
mkdir -p "$SERPER_CACHE_DIR" "$JINA_CACHE_DIR" "$TIKTOKEN_CACHE_DIR"
"${PYTHON:-python3}" -c 'import search_api, sandbox_serper'
exec "${PYTHON:-python3}" -m uvicorn tools.fathom_backend:app \
    --host "${FATHOM_HOST:-127.0.0.1}" --port "${FATHOM_PORT:-8904}" --workers "${FATHOM_WORKERS:-1}" "$@"
