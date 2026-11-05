"""Extract the complete final answer, preserving every subquestion."""

from __future__ import annotations

import re
from typing import Optional

ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _last_boxed_span(text: str) -> Optional[str]:
    """Return the contents of the last `\\boxed{...}`."""
    needle = r"\boxed{"
    start = text.rfind(needle)
    if start < 0:
        return None
    i = start + len(needle)
    depth = 1
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(needle) : i]
        i += 1
    return None  # unbalanced braces


def extract_final_answer(model_output: str) -> str:
    """Extract the final answer block and unwrap a box enclosing the whole answer."""
    visible = re.sub(r"<(think|tool_call)>.*?(?:</\1>|$)", "", model_output, flags=re.DOTALL)
    match = list(ANSWER_BLOCK.finditer(visible))
    if match:
        inner = match[-1].group(1).strip()
        boxed = _last_boxed_span(inner)
        if boxed is not None and inner == "\\boxed{" + boxed + "}":
            return boxed.strip()
        return inner
    return ""
