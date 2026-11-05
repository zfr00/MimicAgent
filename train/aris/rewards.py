import json
import re
import copy
from tools.agent_messages import exemplar_content, text_part, message_content
from tools.image_bank import ImageBank

from evaluation.extract_answer import extract_final_answer
from .config import TRAIN_ROOT
from .protocol import analogy_evidence, format_score, tool_base_score
from .async_utils import gather_strict
from .judging import answer_verdict

FACTORS = {"correct": 1.0, "unknown": 0.5, "incorrect": 0.2}


def combine_rewards(config, verdict, formatting, tool_base, axes):
    if verdict not in FACTORS:
        raise ValueError(f"Invalid verdict {verdict}")
    if set(axes) != {"s_id", "s_map", "s_trans"} or any(type(v) is not int or v not in (0, 1) for v in axes.values()):
        raise ValueError("Analogy judge must return three binary integer scores")
    factor = FACTORS[verdict]
    result = {
        "verdict": verdict, "answer": float(verdict == "correct"),
        "format": float(formatting), "tool_base": tool_base,
        "tool": tool_base * factor, "analogy_base": float(max(axes.values())) if config.analogy_reward_enabled else 0.0,
        "analogy": max(axes.values()) * factor if config.analogy_reward_enabled else 0.0, **axes,
    }
    result["total"] = (config.answer_weight * result["answer"] + config.format_weight * result["format"]
                       + config.tool_weight * result["tool"] + config.analogy_weight * result["analogy"])
    return result


class Rewarder:
    def __init__(self, config, apis):
        self.config, self.apis = config, apis

    async def answer_verdict(self, question, reference, prediction):
        return await answer_verdict(self.config, self.apis, question, reference, prediction)

    async def analogy_axes(self, question, episode):
        zeros = {"s_id": 0, "s_map": 0, "s_trans": 0}
        if not self.config.analogy_reward_enabled:
            return zeros
        exemplars, traces = analogy_evidence(episode)
        if not exemplars or not traces.strip():
            return zeros
        template = (TRAIN_ROOT / "prompts/analogy_reward.txt").read_text(encoding="utf-8")
        parts = []
        for call in episode.calls:
            if call["name"] == "exemplar_recall" and call["success"] and call.get("accepted", True):
                shown = call.get("content")
                if shown is None:
                    parts.extend(exemplar_content(call["result"], image_bank=ImageBank(episode.image_bank,
                                                          max_pixels=self.config.image_max_pixels)))
                else:
                    parts.extend(copy.deepcopy(shown) if isinstance(shown, list) else [text_part(shown)])
        target = [text_part(question)]
        if episode.initial_messages and not self.config.uses_exemplar_prefix:
            shown = episode.initial_messages[-1]["content"]
            target = copy.deepcopy(shown) if isinstance(shown, list) else [text_part(shown)]
        values = {"question": target, "exemplars_block": parts, "traces_block": [text_part(traces)]}
        content = []
        for piece in re.split(r"(\{(?:question|exemplars_block|traces_block)\})", template):
            content.extend(values[piece[1:-1]] if piece.startswith("{") and piece[1:-1] in values else [text_part(piece)])
        content = message_content(content)
        for _ in range(self.config.api_retries):
            reply = await self.apis.chat("analogy", [{"role": "user", "content": content}], 256)
            try:
                axes = json.loads(reply)
                if (isinstance(axes, dict) and set(axes) == set(zeros)
                        and all(type(v) is int and v in (0, 1) for v in axes.values())):
                    return axes
            except json.JSONDecodeError:
                pass
        raise RuntimeError("Invalid analogy reward JSON schema")

    async def score(self, question, reference, episode, *, on_progress=None):
        partial = {"format": format_score(episode),
                   "tool_base": tool_base_score(episode, self.config.tool_similarity_threshold), "total": None}

        def report():
            if "verdict" in partial:
                factor = FACTORS[partial["verdict"]]
                partial["tool"] = partial["tool_base"] * factor
                if "analogy_base" in partial:
                    partial["analogy"] = partial["analogy_base"] * factor
            if on_progress is not None:
                on_progress(dict(partial))

        async def grade_answer():
            verdict = await self.answer_verdict(question, reference, extract_final_answer(episode.final_reply))
            partial.update(verdict=verdict, answer=float(verdict == "correct"))
            report()
            return verdict

        async def grade_analogy():
            axes = await self.analogy_axes(question, episode)
            partial.update(axes)
            partial["analogy_base"] = float(max(axes.values())) if self.config.analogy_reward_enabled else 0.0
            report()
            return axes

        report()
        verdict, axes = await gather_strict(grade_answer(), grade_analogy())
        return combine_rewards(self.config, verdict, partial["format"], partial["tool_base"], axes)
