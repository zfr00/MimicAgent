"""Run one tool request and serialize its HTTP result."""

import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def dispatch(kind, payload):
    if kind == "fathom":
        from sandbox_serper import _safe_eval
        from tools.process_runner import remaining_seconds
        value = _safe_eval(payload["env"], payload["call"], remaining_seconds(350))
        return {"output": "", "result": value,
                "error": value if str(value).startswith("[tool-error]") else None}
    if kind == "kg_backend":
        from neo4j import GraphDatabase
        from openai import OpenAI
        from tools.kg_backend import KGBackend, KGQuery, Settings
        settings = Settings.from_env()
        with GraphDatabase.driver(settings.uri, auth=(settings.username, settings.password)) as driver:
            with OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                        timeout=settings.llm_timeout, max_retries=0) as client:
                return KGBackend(settings, driver, client).retrieve(KGQuery.model_validate(payload))
    from importlib import import_module
    routes = {
        "exemplar_recall": ("exemplar_recall", "ExemplarRecallRequest", "exemplar_recall"),
        "search": ("web_search", "SearchRequest", "search"),
        "query": ("web_search", "QueryRequest", "query"),
        "kg_knowledge": ("kg_knowledge", "KGRequest", "kg_knowledge"),
        "image_crop": ("image_crop", "CropRequest", "image_crop"),
        "code_interpreter": ("code_interpreter", "CodeRequest", "code_interpreter"),
    }
    module_name, model, function = routes[kind]
    module = import_module(f"tools.{module_name}_server")
    return getattr(module, function)(getattr(module, model).model_validate(payload))


def main():
    from fastapi import HTTPException, Response
    from fastapi.encoders import jsonable_encoder
    try:
        with contextlib.redirect_stdout(sys.stderr):
            value = dispatch(sys.argv[1], json.load(sys.stdin))
        if isinstance(value, Response):
            result = {"status_code": value.status_code, "body": json.loads(value.body)}
        else:
            result = {"status_code": 200, "body": jsonable_encoder(value)}
    except HTTPException as exc:
        result = {"status_code": exc.status_code, "body": {"detail": exc.detail}}
    except Exception as exc:
        result = {"status_code": 500, "body": {"error": f"{type(exc).__name__}: {exc}"}}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
