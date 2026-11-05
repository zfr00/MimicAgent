"""Shared student/teacher protocol."""

from dataclasses import dataclass, field
import copy
import json
import re

from tools.agent_messages import question_image_content, exemplar_content, message_content, text_part
from tools.image_bank import ImageBank
from tools.tool_schemas import TOOL_SCHEMAS
from tools.request_models import effective_arguments
from tools.tool_errors import ExemplarPrefixError
from evaluation.extract_answer import extract_final_answer, _last_boxed_span

from .config import DEMO_ROOT
from .tool_similarity import similar_arguments

TOOL_BY_NAME = {tool["name"]: tool for tool in TOOL_SCHEMAS}
ACTION = re.compile(r"\s*<think>(.*?)</think>\s*<(tool_call|answer)>(.*?)</\2>\s*", re.S)
ACTION_BLOCK = re.compile(r"<(tool_call|answer)>(.*?)</\1>", re.S)


@dataclass
class Episode:
    turns: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    stopped_reason: str = ""
    initial_messages: list[dict] = field(default_factory=list)
    image_bank: list[dict] = field(default_factory=list)
    base_messages: list[dict] = field(default_factory=list)

    @property
    def final_reply(self):
        return next((t["content"] for t in reversed(self.turns) if t["role"] == "assistant"), "")

    def as_dict(self):
        return copy.deepcopy({"turns": self.turns, "calls": self.calls, "stopped_reason": self.stopped_reason,
                              "initial_messages": self.initial_messages, "image_bank": self.image_bank,
                              "base_messages": self.base_messages})

    @classmethod
    def from_dict(cls, obj):
        return cls(**obj)


def prompt_template(name):
    text = (DEMO_ROOT / "prompts" / name).read_text(encoding="utf-8")
    match = re.search(r"```\n(.*?)\n```", text, re.S)
    if match is None:
        raise ValueError(f"No fenced prompt in {name}")
    return match.group(1)


def allowed_tool_names(config):
    return {name for name in TOOL_BY_NAME if config.tool_enabled(name)}


def append_exemplar_prefix(messages, result, *, image_bank=None, prefix_k=3):
    """Attach the first K curated companions."""
    if (not isinstance(result, dict) or not tool_succeeded(result)
            or not isinstance(result.get("exemplars"), list) or len(result["exemplars"]) != prefix_k):
        raise RuntimeError(f"Static prefix requires {prefix_k} exemplars")
    content = messages[-1]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    try:
        parts = exemplar_content(result, image_bank=image_bank)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        raise ExemplarPrefixError({'error': str(exc), 'code': 'invalid_prefix'}) from exc
    messages[-1]["content"] = message_content([*content, text_part("\n<exemplars>"),
                                               *parts, text_part("\n</exemplars>")])
    return messages


def system_prompt(config, *, ground_truth=None, question="", image_sizes=()):
    teacher = ground_truth is not None
    schemas = TOOL_SCHEMAS
    remove_recall = config.experiment in {"wo_exemplar_recall", "static_exemplar_prefix"}
    if remove_recall:
        schemas = [tool for tool in schemas if tool["name"] != "exemplar_recall"]
    values = {
        "ground_truth": ground_truth or "",
        "func_schemas": json.dumps(schemas, ensure_ascii=False, indent=2),
        "question": question,
        "width": ", ".join(str(size[0]) for size in image_sizes) or "N/A",
        "height": ", ".join(str(size[1]) for size in image_sizes) or "N/A",
    }
    template = prompt_template("teacher_trajectory_prompt.md" if teacher else "student_system_prompt.md")
    if remove_recall:
        template = "\n".join(line for line in template.splitlines() if "exemplar_recall" not in line)
        if teacher:
            template = template.replace("by solving problems through analogy to retrieved exemplars", "by solving problems step by step")
            template = template.replace(
                "Prefer exemplar-guided analogical reasoning over parametric recall: retrieve a structurally analogous companion exercise, then identify its solution schema, map its roles to the target problem, and transfer its procedure.",
                "Use the supplied companion exercises to identify a solution schema, map its roles to the target problem, and transfer its procedure."
                if config.uses_exemplar_prefix else "Reason from the question, its images, and evidence returned by the available tools.")
            template = template.replace(
                "which retrieved exemplar is structurally analogous, how its schema / roles / procedure transfer to the target, ",
                "how the supplied companions' schema / roles / procedure transfer to the target, "
                if config.uses_exemplar_prefix else "")
    template = template.replace("At most 6 tool-call rounds", f"At most {config.max_tool_rounds} tool-call rounds")
    return re.sub(r"\{(ground_truth|func_schemas|question|width|height)\}",
                    lambda m: values[m.group(1)], template)


def initial_messages(question, image_paths, config, *, ground_truth=None, exemplar_prefix=None,
                     image_bank=None, target_id=None):
    bank = image_bank if image_bank is not None else ImageBank(max_pixels=config.image_max_pixels)
    content, _ = question_image_content(image_paths, question, image_bank=bank, target_id=target_id)
    sizes = []
    if ground_truth is not None:
        from PIL import Image
        for path in image_paths:
            with Image.open(path) as image:
                sizes.append(image.size)
    messages = [{"role": "system", "content": system_prompt(config, ground_truth=ground_truth,
                                                            question=question, image_sizes=sizes)},
                {"role": "user", "content": message_content(content)}]
    return (append_exemplar_prefix(messages, exemplar_prefix, image_bank=bank, prefix_k=config.prefix_k)
            if exemplar_prefix is not None else messages)


def parse_action(reply):
    """Parse one executable action; layout compliance is scored separately."""
    visible = re.sub(r"<think>.*?(?:</think>|$)", "", reply, flags=re.S)
    blocks = list(ACTION_BLOCK.finditer(visible))
    if len(blocks) != 1:
        return None
    kind, body = blocks[0].groups()
    for tag in ("answer", "tool_call"):
        expected = 1 if tag == kind else 0
        if visible.count(f"<{tag}>") != expected or visible.count(f"</{tag}>") != expected:
            return None
    if kind == "answer":
        return (kind, body) if extract_final_answer(reply) else None
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
        return None
    if not isinstance(obj["name"], str) or obj["name"] not in TOOL_BY_NAME or not isinstance(obj["arguments"], dict):
        return None
    from jsonschema import Draft7Validator
    try:
        effective = effective_arguments(obj["name"], obj["arguments"])
    except (ValueError, TypeError, KeyError):
        return None
    if not Draft7Validator(TOOL_BY_NAME[obj["name"]]["parameters"]).is_valid(effective):
        return None
    return kind, obj


def format_score(episode):
    from jsonschema import Draft7Validator
    replies = [t["content"] for t in episode.turns if t["role"] == "assistant"]
    actions = [parse_action(reply) for reply in replies]
    return float(bool(actions) and all(a is not None for a in actions)
                 and all(ACTION.fullmatch(reply) is not None and reply.count("<think>") == 1
                         and reply.count("</think>") == 1 for reply in replies)
                 and all(a[0] == "tool_call" for a in actions[:-1])
                 and all(Draft7Validator(TOOL_BY_NAME[a[1]["name"]]["parameters"]).is_valid(a[1]["arguments"])
                         for a in actions if a[0] == "tool_call")
                 and actions[-1][0] == "answer" and _last_boxed_span(actions[-1][1]) is not None)


def tool_succeeded(result):
    if not isinstance(result, dict):
        return False
    if result.get("error") or result.get("success") is False:
        return False
    return str(result.get("status", "")).lower() not in {"error", "failed", "failure", "timeout"}


def call_fingerprint(name, arguments):
    arguments = effective_arguments(name, arguments)
    for key in ("query", "goal"):
        if name in {"web_search.search", "web_search.query", "kg_knowledge"} and isinstance(arguments.get(key), str):
            arguments[key] = " ".join(arguments[key].split())
    return name, json.dumps(arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tool_base_score(episode, similarity_threshold=0.90):
    n = sum(t["content"].count("<tool_call>") for t in episode.turns if t["role"] == "assistant")
    successful = []
    for call in episode.calls:
        if call["success"]:
            name, encoded = call_fingerprint(call["name"], call["arguments"])
            arguments = json.loads(encoded)
            if not any(name == previous_name and (arguments == previous or
                       similar_arguments(name, arguments, previous, similarity_threshold))
                       for previous_name, previous in successful):
                successful.append((name, arguments))
    return min(len(successful), n) / n if n else 0.0


def analogy_evidence(episode):
    exemplars = []
    first_recall = None
    for call in episode.calls:
        if call["name"] == "exemplar_recall" and call["success"] and call.get("accepted", True):
            exemplars.extend(call["result"].get("exemplars", []))
            first_recall = min(first_recall, call["turn_index"]) if first_recall is not None else call["turn_index"]
    if first_recall is None or not exemplars:
        return [], ""
    traces = []
    for turn in episode.turns[first_recall + 1:]:
        if turn["role"] == "assistant":
            traces.extend(re.findall(r"<think>(.*?)</think>", turn["content"], re.S))
    return exemplars, "\n\n".join(traces)
