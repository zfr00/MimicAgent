"""Per-episode image identities and original pixels, independent of display order."""

import base64
import copy
import hashlib
import io
import json
import math

from PIL import Image


def image_bytes(data_url):
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise ValueError("Expected an embedded image data URL")
    return base64.b64decode(data_url.split(",", 1)[1], validate=True)


def image_part(raw, max_pixels=None):
    with Image.open(io.BytesIO(raw)) as source:
        img = source.convert("RGB")
    size = img.size
    if max_pixels and img.width * img.height > max_pixels:
        scale = math.sqrt(max_pixels / (img.width * img.height))
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url}}, size


class ImageBank:
    def __init__(self, entries=None, max_pixels=262144):
        self.max_pixels = max_pixels
        self.entries = copy.deepcopy(entries or [])
        self._keys = {self._key(e["data_url"], e["source"]): i for i, e in enumerate(self.entries)}
        if any(e["image_index"] != i for i, e in enumerate(self.entries)):
            raise ValueError("Image bank indices must be contiguous and start at zero")

    @staticmethod
    def _key(url, source):
        return hashlib.sha256((json.dumps(source, sort_keys=True) + url).encode()).hexdigest()

    def register(self, raw, source):
        part, (width, height) = image_part(raw)
        url = part["image_url"]["url"]
        key = self._key(url, source)
        if key not in self._keys:
            self._keys[key] = len(self.entries)
            self.entries.append({"image_index": len(self.entries), "source": copy.deepcopy(source),
                                 "data_url": url, "width": width, "height": height})
        return self._keys[key]

    def get(self, index):
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise ValueError(f"Unknown image_index: {index}")
        return self.entries[index]

    def parts(self, index):
        entry = self.get(index)
        part, _ = image_part(image_bytes(entry["data_url"]), self.max_pixels)
        label = (f"\nImage {index} (image_index={index}, "
                 f"source width={entry['width']}, height={entry['height']})\n")
        return [{"type": "text", "text": label}, part]

    def fork(self):
        return ImageBank(self.entries, self.max_pixels)

    def commit(self, candidate):
        if candidate.entries[:len(self.entries)] != self.entries:
            raise ValueError("Image index mismatch")
        self.entries, self._keys = candidate.entries[:], candidate._keys.copy()

    def snapshot(self):
        return copy.deepcopy(self.entries)
