"""Shared executor authentication for training, evaluation and health checks."""
import os


def tool_headers():
    key = os.environ.get("ARIS_TOOL_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}
