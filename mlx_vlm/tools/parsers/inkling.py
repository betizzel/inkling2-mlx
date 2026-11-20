"""Parse an Inkling tool call.

The model writes <|message_model|>NAME<|content_invoke_tool_json|>{"name": NAME,
"args": {...}}<|end_message|>; the JSON body carries the name and an object of
arguments.
"""

from __future__ import annotations

import json
from typing import Any, Optional

tool_call_start: str = "<|content_invoke_tool_json|>"
tool_call_end: str = "<|end_message|>"


def parse_tool_call(text: str, tools: Optional[Any] = None) -> dict:
    call = json.loads(text.strip())
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError(f"Inkling tool call needs a name: {text!r}")
    arguments = call.get("args", call.get("arguments", {}))
    if not isinstance(arguments, dict):
        raise ValueError("Inkling tool arguments must be a JSON object")
    return {"name": call["name"], "arguments": arguments}
