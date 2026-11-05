"""Cancellable request processes with a shared wall-clock deadline."""

import asyncio
import json
import logging
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def request_deadline(request=None, *, timeout=360):
    limit = float(os.environ.get("TOOL_EXECUTION_TIMEOUT", "360"))
    duration = min(float(timeout), limit)
    if request is not None:
        duration = min(duration, float(request.headers.get("X-Tool-Timeout", duration)))
    deadline = time.time() + duration
    inherited = os.environ.get("ARIS_TOOL_DEADLINE_UNIX")
    if inherited:
        deadline = min(deadline, float(inherited))
    if request is not None and request.headers.get("X-Tool-Deadline"):
        deadline = min(deadline, float(request.headers["X-Tool-Deadline"]))
    if not math.isfinite(deadline) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Tool execution timeout must be finite and positive")
    return deadline


def remaining_seconds(default):
    deadline = os.environ.get("ARIS_TOOL_DEADLINE_UNIX")
    remaining = min(default, float(deadline) - time.time()) if deadline else default
    if remaining <= 0:
        raise TimeoutError("Tool execution deadline exceeded")
    return remaining


def backend_headers():
    value = os.environ.get("ARIS_TOOL_DEADLINE_UNIX")
    return {"X-Tool-Deadline": value} if value else {}


async def _read_limited(stream, limit):
    data = bytearray()
    while chunk := await stream.read(65536):
        data.extend(chunk)
        if len(data) > limit:
            raise RuntimeError("Tool worker output exceeds the response limit")
    return bytes(data)


async def _log_stderr(stream):
    remaining = 32768
    while chunk := await stream.read(65536):
        if remaining > 0:
            shown = chunk[:remaining]
            logging.getLogger(__name__).warning("%s", shown.decode("utf-8", errors="replace").rstrip())
            remaining -= len(shown)


async def _disconnected(request):
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(.05)


async def run_job(kind, payload, *, deadline, request=None):
    """Kill the request process group on timeout, disconnect or cancellation."""
    runtime = ROOT / "tools/.runtime"
    runtime.mkdir(exist_ok=True)
    directory = tempfile.TemporaryDirectory(prefix="job-", dir=runtime)
    env = dict(os.environ, ARIS_TOOL_DEADLINE_UNIX=str(deadline), PYTHONDONTWRITEBYTECODE="1",
               ARIS_TOOL_WORK_DIR=directory.name)
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-B", str(ROOT / "tools/process_worker.py"), kind,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env, start_new_session=True)
    except BaseException:
        directory.cleanup()
        raise
    tasks = []
    try:
        async def exchange():
            process.stdin.write(json.dumps(payload, ensure_ascii=False).encode())
            await process.stdin.drain()
            process.stdin.close()
            stdout = await _read_limited(process.stdout, MAX_RESPONSE_BYTES)
            await process.wait()
            if process.returncode:
                raise RuntimeError(f"Tool worker exited with status {process.returncode}")
            return json.loads(stdout)
        result = asyncio.create_task(exchange())
        tasks = [result, asyncio.create_task(_log_stderr(process.stderr))]
        disconnect = asyncio.create_task(_disconnected(request)) if request is not None else None
        if disconnect is not None:
            tasks.append(disconnect)
        done, _ = await asyncio.wait([result] + ([disconnect] if disconnect else []),
                                    timeout=max(0, deadline - time.time()),
                                    return_when=asyncio.FIRST_COMPLETED)
        if result in done:
            return result.result()
        if disconnect in done:
            raise asyncio.CancelledError("Tool caller disconnected")
        raise TimeoutError("Tool execution deadline exceeded")
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        directory.cleanup()


def error_result(exc):
    timeout = isinstance(exc, TimeoutError)
    body = {"error": "Tool execution timed out" if timeout else str(exc),
            "code": "tool_timeout" if timeout else "tool_execution_failed"}
    return {"status_code": 200, "body": body}
