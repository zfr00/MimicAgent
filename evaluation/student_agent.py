"""SciExplore student runner sharing ARIS actions, token budgets and tool dispatch."""

import asyncio
import copy
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import sys

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, REPO_ROOT / 'train'):
    sys.path.insert(0, str(path))

from aris.codec import ChatCodec, TokenBudgetError
from evaluation.budget import EvaluationTranscript
from aris.config import load_config
from aris.interaction import run_episode
from aris.protocol import initial_messages, parse_action, append_exemplar_prefix, allowed_tool_names, system_prompt
from aris.tool_client import dispatch_sync, observation
from tools.agent_messages import MessageContent, with_image_data
from tools.image_bank import ImageBank
from tools.data_records import question_language
from tools.tool_errors import require_exemplar_prefix


@dataclass
class TrajectoryStep:
    role: str
    content: MessageContent
    tool_name: str | None = None
    tool_arguments: dict | None = None


@dataclass
class StudentResult:
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    final_reply: str = ''
    rounds_used: int = 0
    stopped_reason: str = ''
    episode: dict = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    finish: str = 'stop'


def load_codec(checkpoint=None, config=None):
    from transformers import AutoProcessor, AutoTokenizer
    config = config or load_config()
    checkpoint = checkpoint or os.environ.get('ARIS_EVAL_TOKENIZER', 'Qwen/Qwen3-VL-8B-Instruct')
    cache_dir = str(REPO_ROOT / 'cache/tokenizers')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
    return ChatCodec(tokenizer, processor, config.image_max_pixels)


def _render_system_prompt(question):
    return system_prompt(load_config())


def _parse_assistant_reply(reply):
    action = parse_action(reply)
    if action is None:
        return None, None
    return (action[1], None) if action[0] == 'tool_call' else (None, action[1].strip())


def _dispatch(name, arguments, *, image_paths, target_id, target_language="zh", timeout=360,
              image_bank=None, prefix_k=None, retries=3):
    return dispatch_sync(name, arguments, image_paths=image_paths, target_id=target_id,
                         target_language=target_language, timeout=timeout, image_bank=image_bank,
                         prefix_k=prefix_k, retries=retries)


def _client_call(*, base_url, api_key, model, messages, temperature, max_tokens,
                 timeout=180, retries=3, seed=None, api_format='openai', token_parameter='auto',
                 sampling_parameters='auto', client=None):
    from evaluation.model_api import ChatAPI
    owned = client is None
    client = client or ChatAPI(base_url=base_url, api_key=api_key, model=model, api_format=api_format,
                              token_parameter=token_parameter, sampling_parameters=sampling_parameters,
                              timeout=timeout, retries=retries)
    try:
        return client.chat(messages, temperature=temperature, max_tokens=max_tokens, seed=seed)
    finally:
        if owned:
            client.close()


def _resized_messages(messages, image_data):
    return with_image_data(messages, image_data)


async def run_student_async(*, question, image_paths, target_id=None, base_url, api_key, model,
                            temperature=None, max_tokens=None, max_tool_rounds=None,
                            config=None, codec=None, hf_checkpoint=None, seed=None, target_language=None,
                            api_format='openai', token_parameter='auto', sampling_parameters='auto', client=None):
    config = config or load_config()
    target_language = question_language(question, language=target_language)
    if max_tool_rounds is not None:
        config = replace(config, max_tool_rounds=max_tool_rounds).validate()
    temperature = config.eval_temperature if temperature is None else temperature
    if temperature < 0:
        raise ValueError('Evaluation temperature must be nonnegative')
    bank = ImageBank(max_pixels=config.image_max_pixels)
    messages = initial_messages(question, image_paths, config, image_bank=bank, target_id=target_id)
    if config.uses_exemplar_prefix:
        raw = await asyncio.to_thread(_dispatch, "exemplar_recall", {}, image_paths=image_paths,
                                      target_id=target_id, target_language=target_language, timeout=config.tool_timeout,
                                      prefix_k=config.prefix_k, retries=config.tool_retries)
        append_exemplar_prefix(messages, require_exemplar_prefix(json.loads(raw), count=config.prefix_k),
                               image_bank=bank, prefix_k=config.prefix_k)
    student_messages = copy.deepcopy(messages)
    codec = codec or load_codec(hf_checkpoint, config)
    try:
        transcript = EvaluationTranscript(codec, messages, config)
    except TokenBudgetError as exc:
        raise TokenBudgetError(f'{target_id}: {exc}') from exc

    async def generate(transcript):
        budget = min(transcript.remaining, max_tokens) if max_tokens is not None else transcript.remaining
        answer = await asyncio.to_thread(
            _client_call, base_url=base_url, api_key=api_key, model=model,
            messages=_resized_messages(messages, transcript.image_data), temperature=temperature,
            max_tokens=budget, timeout=config.api_timeout, retries=config.api_retries, seed=seed,
            api_format=api_format, token_parameter=token_parameter, sampling_parameters=sampling_parameters, client=client,
        )
        generation = answer if isinstance(answer, Generation) else Generation(answer)
        reply, finish = generation.text, generation.finish
        ids = codec.tokenizer.encode(reply, add_special_tokens=False)
        if not ids:
            raise RuntimeError('Student returned no tokens')
        if len(ids) > transcript.remaining:
            ids = ids[:transcript.remaining]
            reply, finish = codec.tokenizer.decode(ids, skip_special_tokens=False), 'length'
        elif finish != 'length' and len(ids) < transcript.remaining:
            ids.append(codec.eos)
        messages.append({'role': 'assistant', 'content': reply})
        return reply, ids, [0.0] * len(ids), finish

    async def execute(action, staged):
        raw = await asyncio.to_thread(_dispatch, action['name'], action['arguments'],
                                      image_paths=image_paths, target_id=target_id,
                                      target_language=target_language, timeout=config.tool_timeout, image_bank=staged,
                                      retries=config.tool_retries)
        try:
            result = json.loads(raw)
        except ValueError:
            result = {'error': 'Tool returned invalid JSON'}
        return observation(action['name'], result, image_bank=staged, arguments=action['arguments'])

    episode = await run_episode(transcript, generate, execute, config.max_tool_rounds,
                                allowed_tools=allowed_tool_names(config), image_bank=bank,
                                initial_messages=student_messages,
                                on_observation=lambda content: messages.append({'role': 'user', 'content': content}))
    calls = iter(c for c in episode.calls if c.get('accepted', True))
    trajectory = []
    for turn in episode.turns:
        call = next(calls) if turn['role'] == 'tool' else {}
        trajectory.append(TrajectoryStep(**turn, tool_name=call.get('name'), tool_arguments=call.get('arguments')))
    return StudentResult(trajectory, episode.final_reply, len(episode.calls), episode.stopped_reason, episode.as_dict())


def run_student(**kwargs):
    return asyncio.run(run_student_async(**kwargs))


def trajectory_to_dict_list(steps):
    return [asdict(step) for step in steps]
