"""Standalone teacher generation using the same recorded episodes as online ARIS."""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / "train"):
    sys.path.insert(0, str(path))

from aris.clients import Endpoint, ModelAPIs
from aris.config import load_config
from aris.protocol import parse_action, system_prompt
from aris.rollout import Runtime
from aris.tool_client import dispatch_sync, observation
from evaluation.student_agent import load_codec
from tools.agent_messages import MessageContent
from tools.data_records import question_language
from tools.tool_errors import require_exemplar_prefix


@dataclass
class TrajectoryStep:
    role: str
    content: MessageContent
    tool_name: str | None = None
    tool_arguments: dict | None = None


@dataclass
class TeacherResult:
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str | None = None
    rounds_used: int = 0
    stopped_reason: str = ""
    episode: dict = field(default_factory=dict)


def _render_system_prompt(question, ground_truth, width=0, height=0):
    return system_prompt(load_config(), ground_truth=ground_truth, question=question,
                         image_sizes=[(width, height)] if width and height else ())


def _dispatch_tool_call(name, arguments, *, image_paths, target_id, target_language="zh",
                        image_bank=None, timeout=360, prefix_k=None, retries=3):
    return dispatch_sync(name, arguments, image_paths=image_paths, target_id=target_id,
                         target_language=target_language, timeout=timeout, image_bank=image_bank,
                         prefix_k=prefix_k, retries=retries)


def _parse_assistant_reply(reply):
    action = parse_action(reply)
    if action is None:
        return None, None
    return (action[1], None) if action[0] == "tool_call" else (None, action[1].strip())


def _client_call(*, base_url, api_key, model, messages, temperature, max_tokens):
    async def request():
        endpoint = replace(Endpoint.from_env("teacher", model), model=model,
                           base_url=base_url, api_key=api_key)
        apis = ModelAPIs(load_config(), roles=("teacher",), endpoints={"teacher": endpoint})
        try:
            return await apis.chat("teacher", messages, max_tokens, temperature=temperature)
        finally:
            await apis.close()
    return asyncio.run(request())


def run_teacher(*, question, ground_truth, image_paths, target_id=None, target_language=None,
                base_url, api_key, model="gpt-5.2", width=0, height=0, temperature=None,
                max_tokens=4096, max_tool_rounds=6, config=None, codec=None, hf_checkpoint=None):
    config = replace(config or load_config(), max_tool_rounds=max_tool_rounds,
                     teacher_max_tokens_per_turn=max_tokens).validate()
    language = question_language(question, language=target_language)

    class APIs:
        async def chat(self, role, messages, max_tokens):
            return await asyncio.to_thread(_client_call, base_url=base_url, api_key=api_key,
                                           model=model, messages=messages, temperature=temperature,
                                           max_tokens=max_tokens)

    class Tools:
        async def execute(self, action, sample, *, image_bank=None):
            return await self.request(action, sample, image_bank=image_bank)

        async def request(self, action, sample, *, image_bank=None, prefix_k=None):
            raw = await asyncio.to_thread(_dispatch_tool_call, action["name"], action["arguments"],
                                           image_paths=image_paths, target_id=target_id,
                                           target_language=language, image_bank=image_bank,
                                           timeout=config.tool_timeout, retries=config.tool_retries,
                                           **({"prefix_k": prefix_k} if prefix_k is not None else {}))
            try:
                result = json.loads(raw)
            except ValueError:
                result = {"error": "Tool returned invalid JSON"}
            return observation(action["name"], result, image_bank=image_bank, arguments=action["arguments"])

        async def prefetch_exemplars(self, sample):
            result, _, success = await self.request({"name": "exemplar_recall", "arguments": {}}, sample,
                                                    prefix_k=config.prefix_k)
            return require_exemplar_prefix(result, success, count=config.prefix_k)

    runtime = Runtime.__new__(Runtime)
    runtime.prefix_tasks = {}
    runtime.config = config
    runtime.codec = codec or load_codec(hf_checkpoint or os.environ.get('HF_CHECKPOINT'), config)
    runtime.apis, runtime.tools = APIs(), Tools()
    sample = SimpleNamespace(label=ground_truth, metadata={
        "question": question, "target_id": target_id, "language": language, "image_paths": image_paths})
    episode = asyncio.run(runtime.teacher_episode(sample))
    calls = iter(c for c in episode.calls if c.get("accepted", True))
    trajectory = []
    for turn in episode.turns:
        call = next(calls) if turn["role"] == "tool" else {}
        trajectory.append(TrajectoryStep(**turn, tool_name=call.get("name"), tool_arguments=call.get("arguments")))
    _, answer = _parse_assistant_reply(episode.final_reply)
    return TeacherResult(trajectory, answer, len(episode.calls), episode.stopped_reason, episode.as_dict())
