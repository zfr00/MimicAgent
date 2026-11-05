"""Read dataset language and subject."""

from pathlib import Path
import re


def question_language(question, source=None, language=None):
    aliases = {"cn": "zh", "zh": "zh", "en": "en"}
    if language:
        try:
            return aliases[language.lower()]
        except KeyError:
            raise ValueError(f"Unsupported question language: {language}") from None
    if source:
        parts = re.split(r"[^a-z]+", Path(source).stem.lower())
        for part in parts:
            if part in aliases:
                return aliases[part]
    # A Chinese question can contain Latin variables and technical names.
    text = re.sub(r"<[^>]*>|\$[^$]*\$", " ", question)
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    english = len(re.findall(r"[A-Za-z]{2,}", text))
    return "zh" if chinese and chinese >= english else "en"


def question_subject(item, source=None):
    metadata = item.get("metadata") or {}
    if item.get("subject") or metadata.get("subject"):
        return item.get("subject") or metadata["subject"]
    name = Path(source).stem.lower() if source else ""
    for subject in ("physics", "chemistry", "biology"):
        if subject in name:
            return subject
    target_id = str(item.get("id") or metadata.get("target_id") or "").lower()
    match = re.match(r"(?:(?:hs|uni)_)?(phy|chem|bio)_", target_id)
    return {"phy": "physics", "chem": "chemistry", "bio": "biology"}[match[1]] if match else "other"
