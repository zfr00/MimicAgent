"""Independent OpenAI-compatible endpoints for teacher and the three graders."""

import asyncio
from dataclasses import dataclass
import json
import os


class ModelReply(str):
    """Response text with the provider's generation finish reason."""

    def __new__(cls, content, finish_reason="stop"):
        reply = super().__new__(cls, content)
        reply.finish_reason = finish_reason
        return reply


@dataclass(frozen=True)
class Endpoint:
    model: str
    base_url: str
    api_key: str
    token_parameter: str = "max_tokens"

    @classmethod
    def from_env(cls, role, default_model):
        prefix = f"ARIS_{role.upper()}_"
        return cls(
            model=os.environ.get(prefix + "MODEL", default_model),
            base_url=os.environ.get(prefix + "BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            api_key=os.environ.get(prefix + "API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            token_parameter=os.environ.get(prefix + "TOKEN_PARAMETER", "max_completion_tokens" if role == "teacher" else "max_tokens"),
        )


class ModelAPIs:
    def __init__(self, config, roles=None, endpoints=None):
        from openai import AsyncOpenAI
        self.config = config
        self.semaphore = asyncio.Semaphore(config.api_concurrency)
        defaults = {"teacher": "gpt-5.2", "primary": "gpt-4.1-mini",
                    "secondary": "gemini-3-flash", "analogy": "gemini-3-flash"}
        roles = tuple(defaults) if roles is None else tuple(roles)
        overrides = endpoints or {}
        self.endpoints = {role: overrides.get(role) or Endpoint.from_env(role, defaults[role]) for role in roles}
        for role, ep in self.endpoints.items():
            if not ep.api_key:
                raise ValueError(f"Missing ARIS_{role.upper()}_API_KEY (or OPENAI_API_KEY)")
            if ep.token_parameter not in {"max_tokens", "max_completion_tokens"}:
                raise ValueError(f"Unsupported token parameter for {role}")
        self.clients = {role: AsyncOpenAI(api_key=ep.api_key, base_url=ep.base_url,
                                         timeout=config.api_timeout, max_retries=0)
                        for role, ep in self.endpoints.items()}

    async def chat(self, role, messages, max_tokens, *, temperature=None):
        from openai import APIConnectionError, APIStatusError
        ep = self.endpoints[role]
        extra = json.loads(os.environ.get(f"ARIS_{role.upper()}_EXTRA_BODY", "{}"))
        if not isinstance(extra, dict):
            raise ValueError("EXTRA_BODY must be a JSON object")
        options = {} if temperature is None else {"temperature": temperature}
        for attempt in range(self.config.api_retries):
            try:
                async with self.semaphore:
                    response = await self.clients[role].chat.completions.create(
                        model=ep.model, messages=messages,
                        **{ep.token_parameter: max_tokens}, extra_body=extra, **options,
                    )
                choice = response.choices[0] if response.choices else None
                content = choice.message.content if choice is not None else None
                finish = getattr(choice, "finish_reason", "stop")
                if role == "teacher" and finish == "length":
                    return ModelReply(content.strip() if isinstance(content, str) else "", finish)
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(f"{role} returned no visible text; check its token/reasoning budget")
                return ModelReply(content.strip(), finish)
            except (APIConnectionError, APIStatusError, RuntimeError) as exc:
                retryable = (not isinstance(exc, APIStatusError)
                             or exc.status_code in {408, 409, 429} or exc.status_code >= 500)
                if not retryable or attempt + 1 == self.config.api_retries:
                    raise
                await asyncio.sleep(min(0.5 * 2 ** attempt, 4.0))

    async def close(self):
        await asyncio.gather(*(client.close() for client in self.clients.values()))
