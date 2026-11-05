"""Process-managed Fathom search and page extraction."""

import asyncio
import ast
import os
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .process_runner import request_deadline, run_job

app = FastAPI(title="Fathom web backend")
gate = asyncio.Semaphore(int(os.environ.get("MAX_OUTBOUND", "16")))


class Execution(BaseModel):
    env: str
    call: str
    timeout: float = Field(350, gt=0, le=360)


@app.post("/execute")
async def execute(payload: Execution, request: Request):
    try:
        expression = ast.parse(payload.call, mode="eval").body
        if (not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name)
                or expression.func.id not in {"search_urls", "query_url"}
                or payload.env != f"from search_api import {expression.func.id}"):
            raise ValueError("Unsupported Fathom call")
        for argument in [*expression.args, *(x.value for x in expression.keywords)]:
            ast.literal_eval(argument)
    except (SyntaxError, ValueError):
        raise HTTPException(422, "Expected a search_urls or query_url call with literal arguments") from None
    deadline = request_deadline(request, timeout=payload.timeout)
    try:
        async with asyncio.timeout(max(0, deadline - time.time())):
            async with gate:
                result = await run_job("fathom", payload.model_dump(), deadline=deadline, request=request)
        if result["status_code"] != 200:
            raise RuntimeError("Fathom worker failed")
        return result["body"]
    except (TimeoutError, RuntimeError) as exc:
        return {"output": "", "result": "", "error": str(exc) or "Fathom execution timed out"}
