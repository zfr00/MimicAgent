"""Evaluation context budget measured with a shared Qwen3-VL codec."""

from types import SimpleNamespace

from aris.codec import Transcript


class EvaluationTranscript(Transcript):
    def __init__(self, codec, messages, config):
        limit = config.eval_max_context_tokens
        super().__init__(codec, messages, SimpleNamespace(
            max_prompt_tokens=limit, max_response_tokens=limit, max_context_tokens=limit))

    @property
    def remaining(self):
        return self.config.max_context_tokens - len(self.tokens)
