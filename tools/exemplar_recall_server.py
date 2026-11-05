"""Chinese companion lookup with multimodal results and cached English translation."""

import base64
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .request_models import ExemplarRecallRequest
    from .exemplar_translation import TranslationError, translate_exemplars
except ImportError:  # direct script invocation
    from request_models import ExemplarRecallRequest
    from exemplar_translation import TranslationError, translate_exemplars

CURATORIAL_REPO_PATH = os.environ.get(
    "CURATORIAL_REPO_PATH",
    str(Path(__file__).resolve().parent.parent / "data/exemplars/curatorial_repo.jsonl"),
)


class ExemplarImages(BaseModel):
    question: list[str] = Field(default_factory=list)
    answer: list[str] = Field(default_factory=list)
    analysis: list[str] = Field(default_factory=list)


class Exemplar(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = ""
    analysis: str = ""
    knowledge: str = ""
    image: ExemplarImages = Field(default_factory=ExemplarImages)


def _expected_n_companions():
    return 4


def _load_recall_k():
    raw = os.environ.get("EXEMPLAR_RECALL_K", "3")
    try:
        k = int(raw)
    except ValueError:
        raise RuntimeError("EXEMPLAR_RECALL_K must be an integer from 1 to 4") from None
    if not 1 <= k <= _expected_n_companions():
        raise RuntimeError("EXEMPLAR_RECALL_K must be an integer from 1 to 4")
    return k


def _coerce_companion(companion):
    fields = dict(companion)
    if not fields.get("analysis"):
        fields["analysis"] = fields.get("reference_solution") or fields.get("solution") or ""
    return Exemplar.model_validate(fields)


def _load_repo(path):
    by_id = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            target_id = str(entry.get("target_id") or "")
            if not target_id:
                raise RuntimeError(f"{path}:{line_number}: entry must have target_id")
            if target_id in by_id:
                raise RuntimeError(f"{path}:{line_number}: duplicate target_id {target_id}")
            companions = entry.get("companions")
            if not isinstance(companions, list) or len(companions) != _expected_n_companions():
                raise RuntimeError(f"{path}:{line_number}: expected exactly four companions")
            normalized = [_coerce_companion(c).model_dump() for c in companions]
            if len({c['id'] for c in normalized}) != len(normalized):
                raise RuntimeError(f"{path}:{line_number}: duplicate companion id")
            by_id[target_id] = {"target_id": target_id, "companions": normalized}
    return {"by_id": by_id, "count": len(by_id),
            "images_root": str(Path(path).resolve().parent / "images")}


EXEMPLAR_RECALL_K = _load_recall_k()
REPO = _load_repo(CURATORIAL_REPO_PATH) if Path(CURATORIAL_REPO_PATH).is_file() else None
app = FastAPI(title="exemplar_recall")


@app.get("/health")
def health():
    return {"status": "ok" if REPO is not None else "missing_repo",
            "curatorial_repo_path": CURATORIAL_REPO_PATH,
            "n_target_entries": REPO["count"] if REPO else 0,
            "companions_per_target": _expected_n_companions(), "return_k": EXEMPLAR_RECALL_K}


def _images(companions, images_root):
    images = []
    root = Path(images_root).resolve()
    for companion in companions:
        for section in ("question", "answer", "analysis"):
            for index, filename in enumerate(companion["image"][section]):
                path = (root / filename).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise HTTPException(500, f"Exemplar image not found in images/: {filename}")
                images.append({"exemplar_id": companion["id"], "section": section, "index": index,
                               "filename": filename,
                               "image_base64": base64.b64encode(path.read_bytes()).decode("ascii")})
    return images


@app.post("/exemplar_recall")
def exemplar_recall(req: ExemplarRecallRequest):
    if REPO is None:
        raise HTTPException(500, "Place curatorial_repo.jsonl in data/exemplars/ or set CURATORIAL_REPO_PATH")
    entry = REPO["by_id"].get(req.target_id)
    if entry is None:
        raise HTTPException(404, "no curatorial entry registered for this target")
    k = req.prefix_k if req.prefix_k is not None else EXEMPLAR_RECALL_K
    companions = [_coerce_companion(c).model_dump() for c in entry["companions"][:k]]
    images = _images(companions, REPO["images_root"])
    language = req.target_language
    if req.target_language == "en":
        try:
            companions = translate_exemplars(companions)
        except TranslationError as exc:
            language = "zh"
            logging.getLogger(__name__).warning("Exemplar translation: target=%s, returning source text (%s)",
                                                req.target_id, exc)
    return {"exemplars": companions, "images": images, "language": language}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
