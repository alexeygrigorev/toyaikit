import json
from typing import Any, List, Tuple


def _parse_args(arguments: str | dict) -> dict:
    # Expect the SDK to return a JSON string; allow dict for convenience
    if isinstance(arguments, dict):
        return arguments
    return json.loads(arguments)


def find_function_calls_responses(
    messages: List[Any], target_name: str | None = None
) -> List[Tuple[str, dict]]:
    """Extract function calls from OpenAI Responses API outputs.

    Looks for entries with `type == "function_call"` and returns
    (name, parsed_arguments) pairs. Accepts dict or object-shaped items.
    """
    calls: list[tuple[str, dict]] = []
    for m in messages:
        if isinstance(m, dict):
            continue
        if m.type != "function_call":
            continue
        name = m.name
        args_raw = m.arguments
        calls.append((name, _parse_args(args_raw)))

    if target_name is not None:
        calls = [c for c in calls if c[0] == target_name]
    return calls


def find_function_calls_chat_completions(
    messages: List[Any], target_name: str | None = None
) -> List[Tuple[str, dict]]:
    """Extract function calls from Chat Completions assistant messages.

    Looks for assistant messages containing `tool_calls` and returns
    (name, parsed_arguments) pairs. Accepts dict or object-shaped items.
    """
    calls: list[tuple[str, dict]] = []
    for m in messages:
        if isinstance(m, dict):
            continue
        if not m.tool_calls:
            continue
        for tc in m.tool_calls:
            fn = tc.function
            name = fn.name
            args_raw = fn.arguments
            calls.append((name, _parse_args(args_raw)))

    if target_name is not None:
        calls = [c for c in calls if c[0] == target_name]
    return calls


def find_function_calls(messages: List[Any], target_name: str | None = None) -> List[Tuple[str, dict]]:
    """Convenience wrapper that merges responses and chat-completions calls."""
    calls = []
    calls.extend(find_function_calls_responses(messages))
    calls.extend(find_function_calls_chat_completions(messages))
    if target_name is not None:
        calls = [c for c in calls if c[0] == target_name]
    return calls
