"""Inkling tool-call parser (Inkling 2 / Inkling 2 Mini chat format)."""

import json


def test_inkling_tool_parser():
    from mlx_vlm.tools.extractor import process_tool_calls
    from mlx_vlm.tools.parsers import inkling
    from mlx_vlm.tools.registry import _infer_tool_parser

    assert _infer_tool_parser("{{ '<|content_invoke_tool_json|>' }}") == "inkling"
    out = (
        "Checking.<|message_model|>get_weather<|content_invoke_tool_json|>"
        '{"args": {"city": "Athens"}, "name": "get_weather"}<|end_message|>'
    )
    result = process_tool_calls(out, inkling, tools=[])
    assert result.calls[0]["function"]["name"] == "get_weather"
    assert json.loads(result.calls[0]["function"]["arguments"]) == {"city": "Athens"}
    assert result.remaining_text == "Checking."
    plain = out.replace("<|message_model|>", "")
    assert process_tool_calls(plain, inkling, tools=[]).remaining_text == "Checking."
