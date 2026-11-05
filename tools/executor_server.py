"""Unified tools with authentication, admission limits and process deadlines."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import exemplar_recall_server, web_search_server, kg_knowledge_server, image_crop_server, code_interpreter_server
from .process_runner import error_result, request_deadline, run_job
from .request_models import CodeRequest, CropRequest, ExemplarRecallRequest, KGRequest, QueryRequest, SearchRequest

SERVICES = (exemplar_recall_server, web_search_server, kg_knowledge_server, image_crop_server, code_interpreter_server)


@asynccontextmanager
async def lifespan(app):
    if not os.environ.get("ARIS_TOOL_API_KEY"):
        raise RuntimeError("Set ARIS_TOOL_API_KEY for the shared executor")
    concurrency = int(os.environ.get("TOOL_MAX_CONCURRENCY", "128"))
    if concurrency < 1 or float(os.environ.get("TOOL_EXECUTION_TIMEOUT", "360")) <= 0:
        raise ValueError("Tool concurrency and timeout must be positive")
    app.state.gate = asyncio.Semaphore(concurrency)
    yield


app = FastAPI(title="MimicAgent tool executor", lifespan=lifespan)


async def admission(request: Request):
    expected = os.environ.get("ARIS_TOOL_API_KEY", "")
    if not expected or not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {expected}"):
        raise HTTPException(401, "Invalid tool API key")
    gate = request.app.state.gate
    if gate.locked():
        raise HTTPException(429, "Tool concurrency limit reached")
    async with gate:
        yield


async def execute(kind, payload, request):
    try:
        result = await run_job(kind, payload.model_dump(), request=request,
                               deadline=request_deadline(request))
    except (TimeoutError, RuntimeError, ValueError) as exc:
        result = error_result(exc)
    return JSONResponse(status_code=result["status_code"], content=result["body"])


@app.get("/health")
def health():
    status = {}
    for module in SERVICES:
        name = module.__name__.rsplit(".", 1)[-1].removesuffix("_server")
        try:
            status[name] = module.health()
        except Exception:
            status[name] = {"status": "unavailable"}
    return {"status": "ok" if all(v.get("status") == "ok" for v in status.values()) else "unavailable",
            "services": status}


@app.post("/exemplar_recall", dependencies=[Depends(admission)])
async def recall(payload: ExemplarRecallRequest, request: Request):
    return await execute("exemplar_recall", payload, request)


@app.post("/search", dependencies=[Depends(admission)])
async def search(payload: SearchRequest, request: Request):
    return await execute("search", payload, request)


@app.post("/query", dependencies=[Depends(admission)])
async def query(payload: QueryRequest, request: Request):
    return await execute("query", payload, request)


@app.post("/kg_knowledge", dependencies=[Depends(admission)])
async def kg(payload: KGRequest, request: Request):
    return await execute("kg_knowledge", payload, request)


@app.post("/image_crop", dependencies=[Depends(admission)])
async def crop(payload: CropRequest, request: Request):
    return await execute("image_crop", payload, request)


@app.post("/code_interpreter", dependencies=[Depends(admission)])
async def code(payload: CodeRequest, request: Request):
    return await execute("code_interpreter", payload, request)
