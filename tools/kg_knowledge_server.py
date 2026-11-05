"""Adapter from the shared tool executor to the local ATLAS-Wiki backend."""
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from tools.process_runner import backend_headers, remaining_seconds
from fastapi import FastAPI, HTTPException

from tools.request_models import KGRequest

KG_BASE_URL = os.environ.get("KG_BASE_URL", "http://127.0.0.1:10087")
KG_TIMEOUT_SEC = float(os.environ.get("KG_TIMEOUT_SEC", "350"))
app = FastAPI(title="kg_knowledge")


@app.get("/health")
def health():
    try:
        response = requests.get(KG_BASE_URL.rstrip("/") + "/health", timeout=5)
        response.raise_for_status()
        upstream = response.json()
    except (requests.RequestException, ValueError):
        upstream = {"status": "unavailable"}
    return {"status": "ok" if upstream.get("status") in {"ok", "healthy", "ready"} else "unavailable",
            "kg_base_url": KG_BASE_URL, "upstream_health": upstream}


@app.post("/kg_knowledge")
def kg_knowledge(request: KGRequest):
    try:
        response = requests.post(KG_BASE_URL.rstrip("/") + "/tools/kg_knowledge",
                                 json=request.model_dump(), timeout=remaining_seconds(KG_TIMEOUT_SEC) + 1,
                                 headers=backend_headers())
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(502, f"KG backend request failed: {type(exc).__name__}") from exc
    if not isinstance(result, dict) or not all(key in result for key in ("status", "chosen_entity", "triples", "warnings", "conclusion")):
        raise HTTPException(502, "KG backend returned an invalid response")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
