"""Training and evaluation use the same repeated-primary / secondary decision rule."""

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / 'train'):
    sys.path.insert(0, str(path))

from aris.clients import Endpoint, ModelAPIs
from aris.config import load_config
from aris.judging import answer_verdict

Verdict = Literal['correct', 'incorrect', 'unknown']


@dataclass
class JudgeConfig:
    model: str
    base_url: str
    api_key: str
    token_parameter: str = 'max_tokens'

    def endpoint(self):
        return Endpoint(self.model, self.base_url, self.api_key, self.token_parameter)


async def judge_two_stage_async(*, question, ground_truth, model_answer,
                                primary=None, secondary=None, judge_repeats=None,
                                config=None, apis=None):
    config = config or load_config()
    if judge_repeats is not None:
        config = replace(config, judge_repeats=judge_repeats).validate()
    if apis is not None:
        return await answer_verdict(config, apis, question, ground_truth, model_answer)
    endpoints = {}
    if primary is not None:
        endpoints['primary'] = primary.endpoint()
    if secondary is not None:
        endpoints['secondary'] = secondary.endpoint()
    clients = ModelAPIs(config, roles=('primary', 'secondary'), endpoints=endpoints)
    try:
        return await answer_verdict(config, clients, question, ground_truth, model_answer)
    finally:
        await clients.close()


def judge_two_stage(**kwargs):
    return asyncio.run(judge_two_stage_async(**kwargs))


def verdict_to_score(verdict):
    return float(verdict == 'correct')
