"""Python execution with one process, deadline and bounded output per request."""

import os
import hmac
from functools import lru_cache
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.process_runner import remaining_seconds

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from tools.request_models import CodeRequest

TOOL_ROOT = Path(__file__).resolve().parent
EXECUTION_TIMEOUT = float(os.environ.get("CODE_EXECUTION_TIMEOUT", "360"))
MAX_OUTPUT_BYTES = int(os.environ.get("CODE_MAX_OUTPUT_BYTES", "32768"))
if EXECUTION_TIMEOUT <= 0 or MAX_OUTPUT_BYTES <= 0:
    raise ValueError("Python execution timeout and output limit must be positive")


class CodeResponse(BaseModel):
    output: str
    error: str | None = None


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def sandbox_command(source, *, timeout, output_limit):
    binary = os.environ.get("CODE_BWRAP") or shutil.which("bwrap")
    if not binary:
        raise RuntimeError("Install bubblewrap to enable code_interpreter")
    executable = str(Path(sys.executable).resolve())
    command = [binary, "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv", "--dir", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp", "--chdir", "/tmp"]
    paths = [Path("/usr/lib"), Path("/usr/lib64"), Path("/usr/share/fonts"),
             Path("/lib"), Path("/lib64"), Path(sys.base_prefix) / "lib",
             Path(executable)]
    mounted = []
    for path in paths:
        if path.exists() and not any(path == parent or path.is_relative_to(parent) for parent in mounted):
            command += ["--ro-bind", str(path), str(path)]
            mounted.append(path)
    command += ["--ro-bind", sysconfig.get_path("purelib"), "/packages",
                "--ro-bind", str(TOOL_ROOT / "code_worker.py"), "/worker.py",
                "--ro-bind", str(source), "/request.py"]
    environment = {"PATH": "/usr/bin", "LANG": "C.UTF-8", "TMPDIR": "/tmp",
                   "LD_LIBRARY_PATH": str(Path(sys.base_prefix) / "lib"),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/matplotlib",
                   "CODE_CPU_SECONDS": str(max(1, int(timeout) + 1)),
                   "CODE_MEMORY_BYTES": os.environ.get("CODE_MEMORY_BYTES", str(4 * 1024**3)),
                   "CODE_FILE_BYTES": str(max(output_limit, 64 * 1024**2))}
    for key, value in environment.items():
        command += ["--setenv", key, value]
    return command + [executable, "-I", "-B", "-u", "/worker.py", "/request.py"]


def execute_code(code, *, timeout=None, output_limit=None):
    timeout = remaining_seconds(EXECUTION_TIMEOUT if timeout is None else timeout)
    output_limit = MAX_OUTPUT_BYTES if output_limit is None else output_limit
    runtime_dir = Path(os.environ.get("ARIS_TOOL_WORK_DIR", TOOL_ROOT / ".runtime"))
    runtime_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="code-", dir=runtime_dir) as directory:
        source = Path(directory) / "request.py"
        source.write_text(code, encoding="utf-8")
        try:
            command = sandbox_command(source, timeout=timeout, output_limit=output_limit)
        except RuntimeError as exc:
            return CodeResponse(output="", error=str(exc))
        process = subprocess.Popen(
            command, cwd=directory, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        captured = {"stdout": bytearray(), "stderr": bytearray()}
        deadline, error, size = time.monotonic() + timeout, None, 0
        try:
            with selectors.DefaultSelector() as selector:
                for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, name)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        error = f"Code execution timed out after {timeout:g} seconds"
                        break
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fd, 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        room = max(0, output_limit - size)
                        captured[key.data].extend(chunk[:room])
                        size += len(chunk)
                        if size > output_limit:
                            error = f"Output truncated at {output_limit} bytes"
                            break
                    if error:
                        break
            if error is None:
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    error = f"Code execution timed out after {timeout:g} seconds"
        finally:
            _kill_group(process)
            process.wait()
            process.stdout.close()
            process.stderr.close()
        output = captured["stdout"].decode("utf-8", errors="replace")
        stderr = captured["stderr"].decode("utf-8", errors="replace")
        if error is None and process.returncode:
            error = stderr.strip() or f"Python exited with code {process.returncode}"
        return CodeResponse(output=output, error=error)


app = FastAPI(title="code_interpreter")


@lru_cache(maxsize=1)
def check_sandbox():
    command = sandbox_command(TOOL_ROOT / "code_worker.py", timeout=5, output_limit=1024)
    command = command[:-5] + ["-I", "-B", "-c",
        "import sys; sys.path.insert(0, '/packages'); import numpy, scipy, sympy, matplotlib, pandas"]
    result = subprocess.run(command, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, timeout=10)
    if result.returncode:
        raise RuntimeError("bubblewrap requires enabled Linux user and network namespaces")


@app.get("/health")
def health():
    try:
        check_sandbox()
        return {"status": "ok", "timeout": EXECUTION_TIMEOUT, "max_output_bytes": MAX_OUTPUT_BYTES}
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"status": "sandbox_unavailable", "error": str(exc)}


def authenticate(request: Request):
    key = os.environ.get("ARIS_TOOL_API_KEY", "")
    if not key or not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {key}"):
        raise HTTPException(401, "Invalid tool API key")


@app.post("/code_interpreter", response_model=CodeResponse, dependencies=[Depends(authenticate)])
def code_interpreter(req: CodeRequest):
    return execute_code(req.code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
