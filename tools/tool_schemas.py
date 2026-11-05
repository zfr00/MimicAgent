from __future__ import annotations

import json
from typing import Any, Dict, List


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "exemplar_recall",
        "description": (
            "Retrieve structurally analogous companion exercises for the current "
            "problem from the curatorial exemplar repository attached to "
            "SciExplore-Bench. The current problem "
            "is selected automatically. The companions are pre-aligned by domain "
            "experts in fixed curatorial order. Each exemplar is returned with "
            "its problem statement and a "
            "reference solution trace, so the agent can transfer the solution "
            "schema (structure -> role mapping -> procedure) to the target problem."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "web_search.search",
        "description": (
            "Open-domain keyword search. Returns a ranked list of candidate URLs "
            "with short snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search.query",
        "description": (
            "Visit a single URL (must come from a previous web_search.search "
            "result) and extract evidence relevant to the stated goal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "goal": {
                    "type": "string",
                    "description": "What to extract: definitions, formulae, boundary conditions, etc.",
                },
            },
            "required": ["url", "goal"],
        },
    },
    {
        "name": "kg_knowledge",
        "description": (
            "Multi-hop lookup over the ATLAS-Wiki knowledge graph; returns "
            "structured evidence (subject-relation-object triples) plus a short "
            "natural-language conclusion. Use for concept definitions, entity "
            "relations, attributes, and causal chains."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language question; one sentence works best.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "image_crop",
        "description": (
            "Crop a region of any image already shown (question, exemplar, or earlier crop), using its image_index, and re-inject the crop "
            "as multimodal context, for inspecting ticks, labels, small text, or "
            "circuit connections. bbox uses normalized coordinates by default "
            "(origin at top-left, x to the right, y down)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "Stable 0-based image_index shown beside the image in this conversation.",
                },
                "bbox": {
                    "type": "object",
                    "description": "Crop box in normalized coordinates (default) or source-image pixels (bbox_type='px'); use the labeled source dimensions.",
                    "properties": {
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                        "x2": {"type": "number"},
                        "y2": {"type": "number"},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                },
                "bbox_type": {"type": "string", "default": "norm", "enum": ["norm", "px"]},
                "pad": {"type": "number", "default": 0.0},
            },
            "required": ["image_index", "bbox"],
        },
    },
    {
        "name": "code_interpreter",
        "description": (
            "Execute Python in a sandbox with numpy / scipy / sympy / matplotlib "
            "/ pandas pre-loaded. Use it for equation solving, unit conversion, "
            "and numeric verification. Each call runs in a fresh Python process."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code; multi-line via \\n."},
            },
            "required": ["code"],
        },
    },
]


def schemas_json(indent: int = 2) -> str:
    """Render the schema list as the JSON string consumed by ``{func_schemas}``."""
    return json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=indent)


TOOL_ENDPOINTS: Dict[str, Dict[str, str]] = {
    "exemplar_recall":   {"url": "http://127.0.0.1:8000/exemplar_recall"},
    "web_search.search": {"url": "http://127.0.0.1:8000/search"},
    "web_search.query":  {"url": "http://127.0.0.1:8000/query"},
    "kg_knowledge":      {"url": "http://127.0.0.1:8000/kg_knowledge"},
    "image_crop":        {"url": "http://127.0.0.1:8000/image_crop"},
    "code_interpreter":  {"url": "http://127.0.0.1:8000/code_interpreter"},
}


if __name__ == "__main__":
    print(schemas_json())
