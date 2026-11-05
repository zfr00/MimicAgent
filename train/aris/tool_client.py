"""Shared tool dispatch contract for training, evaluation and teacher runners."""

import asyncio
import json
import os
import time

import aiohttp
import requests

from tools.agent_messages import tool_response_content
from tools.tool_schemas import TOOL_ENDPOINTS
from tools.data_records import question_language
from tools.tool_errors import require_exemplar_prefix
from tools.http_config import tool_headers
from .protocol import tool_succeeded


def tool_endpoints():
    endpoints = {name: spec["url"] for name, spec in TOOL_ENDPOINTS.items()}
    overrides = json.loads(os.environ.get("ARIS_TOOL_ENDPOINTS", "{}"))
    if set(overrides) - set(endpoints):
        raise ValueError("ARIS_TOOL_ENDPOINTS contains an unknown tool name")
    endpoints.update(overrides)
    return endpoints


def tool_payload(name, arguments, metadata, image_bank=None, *, prefix_k=None):
    if name == "exemplar_recall":
        if not metadata.get("target_id"):
            raise ValueError("exemplar_recall requires the current problem's target_id")
        payload = {"target_id": metadata["target_id"],
                "target_language": question_language(metadata.get("question", ""),
                                                      language=metadata.get("language") or (None if metadata.get("question") else "zh"))}
        if prefix_k is not None:
            payload["prefix_k"] = prefix_k
        return payload
    payload = dict(arguments)
    if name == "image_crop":
        payload.pop("image_data", None)
        if image_bank is None:
            payload["image_paths"] = metadata["image_paths"]
        else:
            payload["image_paths"] = []
            payload["image_data"] = image_bank.get(payload.get("image_index"))["data_url"]
    return payload


def observation(name, result, *, image_bank=None, arguments=None):
    if not isinstance(result, dict):
        result = {"error": "Tool returned non-object JSON"}
    content = tool_response_content(name, json.dumps(result, ensure_ascii=False),
                                    image_bank=image_bank, arguments=arguments)
    success = tool_succeeded(result)
    if name == "image_crop":
        success = success and isinstance(content, list)
    elif name == "exemplar_recall" and result.get("images"):
        success = success and isinstance(content, list)
    if isinstance(content, str) and name in {"exemplar_recall", "image_crop"}:
        try:
            error = json.loads(content.removeprefix("<tool_response>").removesuffix("</tool_response>"))["data"].get("error")
        except (ValueError, TypeError, KeyError, AttributeError):
            error = None
        if error:
            result = {**result, "error": error}
            success = False
    return result, content, success


class ToolExecutor:
    def __init__(self, session, config):
        self.session, self.config, self.endpoints = session, config, tool_endpoints()

    async def execute(self, action, sample, *, image_bank=None):
        if not self.config.tool_enabled(action["name"]):
            return observation(action["name"], {"error": f"Tool disabled in {self.config.experiment}"})
        return await self._request(action, sample, image_bank=image_bank)

    async def prefetch_exemplars(self, sample):
        result, _, success = await self._request({"name": "exemplar_recall", "arguments": {}}, sample,
                                                prefix_k=self.config.prefix_k)
        return require_exemplar_prefix(result, success, count=self.config.prefix_k)

    async def _request(self, action, sample, *, image_bank=None, prefix_k=None):
        name = action["name"]
        try:
            payload = tool_payload(name, action["arguments"], sample.metadata, image_bank, prefix_k=prefix_k)
        except ValueError as exc:
            return observation(name, {"error": str(exc), "code": "invalid_tool_request"})
        deadline = time.monotonic() + self.config.tool_timeout
        result = {"error": "Tool request timed out", "code": "tool_timeout"}
        for attempt in range(self.config.tool_retries):
            remaining = self.config.tool_timeout if attempt == 0 else deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                async with self.session.post(self.endpoints[name], json=payload,
                                             headers={**tool_headers(), "X-Tool-Timeout": str(remaining)},
                                             timeout=aiohttp.ClientTimeout(total=remaining + 5)) as response:
                    result = await response.json()
                    response.raise_for_status()
                retry = isinstance(result, dict) and result.get("code") in {"tool_timeout", "tool_execution_failed"}
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                result = {"error": f"{name}: {type(exc).__name__}: {exc}", "code": "tool_request_failed"}
                status = getattr(exc, "status", 0)
                retry = status not in range(400, 500) or status in {408, 429}
            if not retry or attempt + 1 == self.config.tool_retries:
                break
            await asyncio.sleep(min(.5 * 2 ** attempt, max(0, deadline - time.monotonic())))
        return observation(name, result, image_bank=image_bank, arguments=action["arguments"])


def dispatch_sync(name, arguments, *, image_paths, target_id, target_language="zh", timeout=360,
                  image_bank=None, prefix_k=None, retries=3):
    endpoints = tool_endpoints()
    if name not in endpoints:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        payload = tool_payload(name, arguments, {"image_paths": image_paths, "target_id": target_id,
                                               "language": target_language}, image_bank, prefix_k=prefix_k)
    except ValueError as exc:
        return json.dumps({"error": str(exc), "code": "invalid_tool_request"})
    deadline = time.monotonic() + timeout
    result = {"error": "Tool request timed out", "code": "tool_timeout"}
    for attempt in range(retries):
        remaining = timeout if attempt == 0 else deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            response = requests.post(endpoints[name], json=payload, timeout=remaining + 5,
                                     headers={**tool_headers(), "X-Tool-Timeout": str(remaining)})
            result = json.loads(response.text)
            response.raise_for_status()
            retry = isinstance(result, dict) and result.get("code") in {"tool_timeout", "tool_execution_failed"}
        except (requests.RequestException, ValueError) as exc:
            result = {"error": f"{name}: {type(exc).__name__}: {exc}", "code": "tool_request_failed"}
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            retry = status not in range(400, 500) or status in {408, 429}
        if not retry or attempt + 1 == retries:
            break
        time.sleep(min(.5 * 2 ** attempt, max(0, deadline - time.monotonic())))
    return json.dumps(result, ensure_ascii=False)
