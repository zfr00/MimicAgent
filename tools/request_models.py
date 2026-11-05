"""Effective HTTP request fields shared by executors and reward deduplication."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExemplarRecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str = Field(..., min_length=1)
    target_language: Literal["zh", "en"] = "zh"
    prefix_k: int | None = Field(default=None, ge=1, le=4, strict=True)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)


class QueryRequest(BaseModel):
    url: str
    goal: str


class KGRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(10, ge=1, le=30)
    hops: int = Field(2, ge=1, le=4)
    limit_per_hop: int = Field(5, ge=1, le=20)
    fuzzy_limit: int = Field(10, ge=1, le=30)


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class CropRequest(BaseModel):
    image_index: int = Field(..., ge=0)
    image_paths: list[str] = Field(default_factory=list)
    image_data: str | None = None  # runner supplies one image from the episode bank
    bbox: BBox
    bbox_type: Literal["norm", "px"] = "norm"
    pad: float = 0.0
    out_format: Literal["png", "jpeg"] = "png"


class CodeRequest(BaseModel):
    code: str


REQUEST_MODELS = {
    "web_search.search": SearchRequest,
    "web_search.query": QueryRequest,
    "kg_knowledge": KGRequest,
    "image_crop": CropRequest,
    "code_interpreter": CodeRequest,
}


def effective_arguments(name, arguments):
    """Normalize tool request arguments."""
    if name == "exemplar_recall":
        return {}
    payload = dict(arguments)
    if name == "image_crop":
        payload["image_paths"] = []
        payload["image_data"] = None
    result = REQUEST_MODELS[name].model_validate(payload).model_dump()
    result.pop("image_paths", None)
    result.pop("image_data", None)
    return result
