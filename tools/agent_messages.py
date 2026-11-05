"""Shared inline image rendering for students, teachers, prefixes and rewards."""

import base64
import binascii
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .image_bank import ImageBank

MessageContent = Union[str, List[Dict[str, Any]]]


def text_part(text):
    return {"type": "text", "text": text}


def interleave_images(text, images, *, label):
    """Consume each field's placeholders in order; unmarked images follow text."""
    if not isinstance(text, str):
        raise ValueError(f"{label}: expected text")
    count = text.count("<image>")
    if count and count != len(images):
        raise ValueError(f"{label}: {count} <image> placeholders for {len(images)} images")
    if not count:
        return ([text_part(text)] if text else []) + [part for image in images for part in image]
    parts = []
    segments = text.split("<image>")
    for i, segment in enumerate(segments):
        if segment:
            parts.append(text_part(segment))
        if i < len(images):
            parts.extend(images[i])
    return parts


def message_content(parts):
    return parts if any(p["type"] == "image_url" for p in parts) else "".join(p["text"] for p in parts)


def question_image_content(image_paths, question="", *, image_bank=None, target_id=None):
    bank = image_bank if image_bank is not None else ImageBank()
    staged, images, sizes = bank.fork(), [], []
    for position, path in enumerate(image_paths):
        index = staged.register(Path(path).read_bytes(),
                                {"kind": "question", "target_id": target_id, "position": position})
        entry = staged.get(index)
        images.append(staged.parts(index))
        sizes.append((entry["width"], entry["height"]))
    content = interleave_images(question, images, label=f"{target_id or 'target'}.question")
    bank.commit(staged)
    return content, sizes


def exemplar_content(result, *, image_bank=None):
    """Render exemplar text and images."""
    bank = image_bank if image_bank is not None else ImageBank()
    staged = bank.fork()
    records = {}
    for record in result.get("images", []):
        key = (record["exemplar_id"], record["section"], record["index"])
        if key in records:
            raise ValueError(f"Duplicate exemplar image: {key}")
        records[key] = record
    used, parts = set(), []
    exemplars = result.get("exemplars")
    if not isinstance(exemplars, list) or not exemplars:
        raise ValueError("Recall returned no exemplars")
    for number, exemplar in enumerate(exemplars, 1):
        eid = exemplar["id"]
        parts.append(text_part(f"\nExemplar {number} ({eid})\n"))
        for section in ("question", "answer", "analysis"):
            chunks = []
            names = (exemplar.get("image") or {}).get(section, [])
            for position, filename in enumerate(names):
                key = (eid, section, position)
                record = records.get(key)
                if record is None or record["filename"] != filename:
                    raise ValueError(f"Missing or mismatched image: {eid}.{section}[{position}]")
                used.add(key)
                index = staged.register(base64.b64decode(record["image_base64"], validate=True),
                                        {"kind": "exemplar", "exemplar_id": eid,
                                         "section": section, "position": position})
                chunks.append(staged.parts(index))
            parts.append(text_part(f"\n{section.capitalize()}:\n"))
            parts.extend(interleave_images(exemplar.get(section, ""), chunks, label=f"{eid}.{section}"))
        parts.append(text_part(f"\nKnowledge: {exemplar.get('knowledge', '')}\n"))
    if used != set(records):
        raise ValueError("Unmatched exemplar image attachments")
    bank.commit(staged)
    return parts


def _tool_response_block(name, result_json_text):
    return f'<tool_response>{{"tool":{json.dumps(name)},"data":{result_json_text}}}</tool_response>'


def tool_response_content(name, result_json_text, *, image_bank=None, arguments=None):
    if name not in {"image_crop", "exemplar_recall"}:
        return _tool_response_block(name, result_json_text)
    try:
        data = json.loads(result_json_text)
    except (ValueError, TypeError):
        return _tool_response_block(name, result_json_text)
    if not isinstance(data, dict) or data.get("error") or "detail" in data:
        return _tool_response_block(name, result_json_text)
    bank = image_bank if image_bank is not None else ImageBank()
    staged = bank.fork()
    try:
        if name == "exemplar_recall":
            parts = [text_part('<tool_response>\nTool: exemplar_recall\n'),
                     *exemplar_content(data, image_bank=staged), text_part("\n</tool_response>")]
        else:
            if "image_base64" not in data:
                return _tool_response_block(name, result_json_text)
            source = data.get("source_image_index", (arguments or {}).get("image_index"))
            bbox = data.get("bbox", (arguments or {}).get("bbox"))
            provenance = {"kind": "crop", "source_image_index": source, "bbox": bbox,
                          "bbox_type": data.get("bbox_type", (arguments or {}).get("bbox_type", "norm"))}
            index = staged.register(base64.b64decode(data["image_base64"], validate=True), provenance)
            description = '<tool_response>\nTool: image_crop\nCrop result.'
            if source is not None:
                description += f" Source image_index={source}."
            if bbox is not None:
                description += f" Region ({provenance['bbox_type']}): {json.dumps(bbox)}."
            parts = [text_part(description), *staged.parts(index), text_part("\n</tool_response>")]
    except (binascii.Error, OSError, TypeError, ValueError, KeyError) as exc:
        return _tool_response_block(name, json.dumps({"error": f"invalid {'crop' if name == 'image_crop' else 'exemplar'} image content: {exc}"}))
    bank.commit(staged)
    return message_content(parts)


def with_image_data(messages, image_data):
    """Replace each displayed occurrence, including repetitions of one image index."""
    result = copy.deepcopy(messages)
    parts = [p for m in result if isinstance(m["content"], list)
             for p in m["content"] if p["type"] == "image_url"]
    if len(parts) != len(image_data):
        raise ValueError("Message image occurrences and encoded images disagree")
    for part, url in zip(parts, image_data, strict=True):
        part["image_url"]["url"] = url
    return result
