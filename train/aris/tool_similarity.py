"""Near-duplicate matching for natural-language tool arguments."""

from collections import Counter
from difflib import SequenceMatcher
import re
import unicodedata


TEXT_FIELDS = {"web_search.search": "query", "web_search.query": "goal", "kg_knowledge": "query"}
WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u3400-\u9fff]|\d+(?:\.\d+)?")
FILLER = set("a an the is are of for to please find explain describe how what does do".split())
UNITS = set("m cm mm km s ms min h kg g mg mol mmol K C F Pa kPa MPa J kJ W kW V mV A mA N Hz kHz L mL eV".split())
NEGATIONS = re.compile(r"\b(?:not|no|without|increase|decrease|positive|negative)\b|不|无|非|增|减|正|负", re.I)


def _text_features(text):
    text = unicodedata.normalize("NFC", text)
    words = WORDS.findall(text)
    protected = (
        tuple(re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)),
        tuple(re.findall(r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", text, re.S)),
        tuple(re.findall(r"\b[A-Za-z]\w*\s*[=^/*]\s*[A-Za-z\d()+-]+", text)),
        tuple(sorted(w for w in words if w in UNITS or re.fullmatch(r"(?:[A-Z][a-z]?)+", w)
                     or (re.fullmatch(r"[a-z]", w) and w != 'a'))),
        tuple(re.findall(r"[<>=≤≥≠+−±^%/*-]", text)),
        tuple(sorted(m.casefold() for m in NEGATIONS.findall(text))),
    )
    tokens = [word.casefold() for word in words if word.casefold() not in FILLER]
    return protected, tokens


def similar_arguments(name, left, right, threshold):
    field = TEXT_FIELDS.get(name)
    if field is None or {k: v for k, v in left.items() if k != field} != {
            k: v for k, v in right.items() if k != field}:
        return False
    protected_left, tokens_left = _text_features(left[field])
    protected_right, tokens_right = _text_features(right[field])
    if protected_left != protected_right or not tokens_left or not tokens_right:
        return False
    if tokens_left == tokens_right:
        return True
    a, b = " ".join(tokens_left), " ".join(tokens_right)
    if min(len(a), len(b)) < 8:
        return False
    overlap = sum((Counter(tokens_left) & Counter(tokens_right)).values())
    word_score = 2 * overlap / (len(tokens_left) + len(tokens_right))
    char_score = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return (char_score if re.search(r"[\u3400-\u9fff]", a + b) else max(word_score, char_score)) >= threshold
