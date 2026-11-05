"""One dual-judge decision rule for training and evaluation."""

import re

from .async_utils import gather_strict
from .protocol import prompt_template

VERDICTS = {"correct", "incorrect", "unknown"}


async def answer_verdict(config, apis, question, reference, prediction):
    if not prediction.strip():
        return "unknown"
    values = {"question": question, "ground_truth": reference, "model_answer": prediction}
    prompt = re.sub(r"\{(question|ground_truth|model_answer)\}", lambda m: values[m.group(1)],
                    prompt_template("answer_judge_prompt.md"))
    messages = [{"role": "user", "content": prompt}]
    replies = await gather_strict(*(apis.chat("primary", messages, 64)
                                    for _ in range(config.judge_repeats)))
    verdicts = [reply.strip().lower() for reply in replies]
    if len(set(verdicts)) == 1 and verdicts[0] in {"correct", "incorrect"}:
        return verdicts[0]
    result = (await apis.chat("secondary", messages, 64)).strip().lower()
    return result if result in VERDICTS else "unknown"
