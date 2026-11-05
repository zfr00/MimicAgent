"""Local validation of question and exemplar images and their inline placeholders."""

from pathlib import Path

from PIL import Image


def validate_images(text, paths, *, label, checked=None):
    checked = checked if checked is not None else set()
    count = text.count('<image>')
    if count and count != len(paths):
        raise ValueError(f'{label}: {count} <image> placeholders for {len(paths)} images')
    for name in paths:
        path = Path(name).resolve()
        if path in checked:
            continue
        if not path.is_file():
            raise FileNotFoundError(f'{label}: image not found: {path}')
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError) as exc:
            raise ValueError(f'{label}: Invalid image encoding: {path}') from exc
        checked.add(path)
