from dataclasses import dataclass, fields
import json
import os
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = TRAIN_ROOT.parent
EXPERIMENTS = ("full", "wo_tools", "prefix", "wo_exemplar_recall", "static_exemplar_prefix", "wo_analogy_reward")


@dataclass(frozen=True)
class Config:
    experiment: str = "full"
    prefix_k: int = 3
    group_size: int = 8
    prompts_per_step: int = 8
    imitation_lambda: float = 10.0
    temperature: float = 0.8
    eval_temperature: float = 0.8
    eval_runs: int = 5
    eval_max_context_tokens: int = 25600
    clip_low: float = 0.20
    clip_high: float = 0.28
    advantage_epsilon: float = 1e-6
    advantage_std_correction: int = 1
    max_tool_rounds: int = 6
    max_prompt_tokens: int = 3600
    max_response_tokens: int = 7680
    max_context_tokens: int = 10240
    image_max_pixels: int = 262144
    tool_timeout: float = 360.0
    tool_retries: int = 3
    tool_similarity_threshold: float = 0.90
    api_timeout: float = 180.0
    api_retries: int = 3
    rollout_concurrency: int = 160
    api_concurrency: int = 16
    teacher_attempts: int = 3
    teacher_max_tokens_per_turn: int = 4096
    judge_repeats: int = 3
    answer_weight: float = 0.40
    format_weight: float = 0.15
    tool_weight: float = 0.25
    analogy_weight: float = 0.20
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0

    @property
    def uses_exemplar_prefix(self):
        return self.experiment in {"prefix", "static_exemplar_prefix"}

    def tool_enabled(self, name):
        if self.experiment in {"wo_tools", "prefix"}:
            return False
        return name != "exemplar_recall" or self.experiment not in {"wo_exemplar_recall", "static_exemplar_prefix"}

    @property
    def analogy_reward_enabled(self):
        return self.experiment != "wo_analogy_reward" and self.analogy_weight > 0

    @property
    def training_api_roles(self):
        roles = ("teacher", "primary", "secondary")
        if self.analogy_reward_enabled and self.tool_enabled("exemplar_recall"):
            roles += ("analogy",)
        return roles

    def validate(self):
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {self.experiment}")
        if type(self.prefix_k) is not int or not 1 <= self.prefix_k <= 4:
            raise ValueError("prefix_k must be an integer from 1 to 4")
        for name in ("group_size", "prompts_per_step", "max_prompt_tokens", "max_response_tokens",
                     "max_context_tokens", "api_retries", "rollout_concurrency", "api_concurrency",
                     "teacher_attempts", "teacher_max_tokens_per_turn", "judge_repeats", "image_max_pixels", "eval_runs",
                     "eval_max_context_tokens", "tool_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.group_size < 2 or self.max_tool_rounds < 0:
            raise ValueError("ARIS needs group_size >= 2 and max_tool_rounds >= 0")
        if self.advantage_std_correction not in (0, 1):
            raise ValueError("advantage_std_correction must be 0 or 1")
        if not 0 < self.temperature or not 0 < self.imitation_lambda:
            raise ValueError("temperature and imitation_lambda must be positive")
        if self.eval_temperature < 0:
            raise ValueError("eval_temperature must be nonnegative")
        if self.tool_timeout <= 0 or self.api_timeout <= 0:
            raise ValueError("Tool and API timeouts must be positive")
        if not 0 < self.tool_similarity_threshold <= 1:
            raise ValueError("tool_similarity_threshold must be in (0, 1]")
        if not 0 < self.clip_low < 1 or self.clip_high <= 0:
            raise ValueError("Invalid asymmetric clipping bounds")
        if self.max_prompt_tokens >= self.max_context_tokens:
            raise ValueError("max_prompt_tokens must leave room for a response")
        weights = (self.answer_weight, self.format_weight, self.tool_weight, self.analogy_weight)
        if min(weights) < 0 or abs(sum(weights) - 1) > 1e-9:
            raise ValueError("Reward weights must be nonnegative and sum to one")
        return self


def load_config(path=None):
    path = path or os.environ.get("ARIS_CONFIG") or TRAIN_ROOT / "configs/aris.json"
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown ARIS settings: {sorted(unknown)}")
    return Config(**values).validate()
