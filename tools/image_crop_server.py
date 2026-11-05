"""Crop and resample an image region for multimodal feedback."""

from __future__ import annotations

import base64
import io
import math
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel


try:
    from .request_models import BBox, CropRequest
    from .image_bank import image_bytes
except ImportError:  # direct script invocation
    from request_models import BBox, CropRequest
    from image_bank import image_bytes


class CropResponse(BaseModel):
    image_base64: str
    width: int
    height: int
    note: Optional[str] = None
    source_image_index: int
    bbox: list[float]
    bbox_type: str = "norm"


app = FastAPI(title="image_crop")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/image_crop", response_model=CropResponse)
def image_crop(req: CropRequest) -> CropResponse:
    if req.image_data is not None:
        try:
            with Image.open(io.BytesIO(image_bytes(req.image_data))) as source:
                img = source.convert("RGB")
        except (ValueError, OSError, IndexError) as exc:
            raise HTTPException(400, "Invalid image supplied by the runner") from exc
    else:
        if req.image_index >= len(req.image_paths):
            raise HTTPException(400, f"image_index out of range: {req.image_index}")
        with Image.open(req.image_paths[req.image_index]) as source:
            img = source.convert("RGB")
    W, H = img.size

    # Resolve bbox into pixel coords.
    if req.bbox_type == "norm":
        x1, y1, x2, y2 = req.bbox.x1 * W, req.bbox.y1 * H, req.bbox.x2 * W, req.bbox.y2 * H
    else:
        x1, y1, x2, y2 = req.bbox.x1, req.bbox.y1, req.bbox.x2, req.bbox.y2

    if req.pad > 0:
        px = req.pad * W
        py = req.pad * H
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(W, int(round(x2)))
    y2 = min(H, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(400, "degenerate bbox after clipping")

    crop = img.crop((x1, y1, x2, y2))
    max_pixels = int(os.environ.get("CROP_MAX_PIXELS", "262144"))
    scale = min(2.0, math.sqrt(max_pixels / (crop.width * crop.height)))
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format=req.out_format.upper())
    return CropResponse(
        image_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
        width=crop.width,
        height=crop.height,
        source_image_index=req.image_index,
        bbox=[x1 / W, y1 / H, x2 / W, y2 / H],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
