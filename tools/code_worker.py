"""Per-request worker for code_interpreter_server."""

import importlib
import os
from pathlib import Path
import resource
import sys
import traceback


def make_namespace():
    namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
    for name in ("math", "cmath", "fractions", "decimal", "statistics", "random",
                 "itertools", "functools", "re", "json"):
        namespace[name] = importlib.import_module(name)
    namespace.update(Fraction=namespace["fractions"].Fraction, Decimal=namespace["decimal"].Decimal)
    for name, alias in (("numpy", "np"), ("sympy", "sp"), ("scipy", None),
                        ("pandas", "pd"), ("matplotlib", None), ("matplotlib.pyplot", "plt")):
        module = importlib.import_module(name)
        namespace[name.split(".")[0]] = importlib.import_module(name.split(".")[0])
        if alias:
            namespace[alias] = module
    return namespace


def main():
    try:
        sys.path.insert(0, "/packages")
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        for limit, variable in ((resource.RLIMIT_CPU, "CODE_CPU_SECONDS"),
                                (resource.RLIMIT_AS, "CODE_MEMORY_BYTES"),
                                (resource.RLIMIT_FSIZE, "CODE_FILE_BYTES")):
            value = int(os.environ[variable])
            resource.setrlimit(limit, (value, value))
        source = Path(sys.argv[1]).read_text(encoding="utf-8")
        namespace = make_namespace()
        exec(compile(source, "<tool code>", "exec"), namespace, namespace)
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
