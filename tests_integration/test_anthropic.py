import os

import pytest
from pydantic import BaseModel

from toyaikit import AnthropicClient, AnthropicMessagesRunner
from toyaikit.tools import Tools

from .utils import _TestCallback


def get_anthropic_client():
    """Create an AnthropicClient using environment variables."""
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key or not base_url:
        pytest.skip("ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL not set")

    return AnthropicClient(
        api_key=api_key,
        base_url=base_url,
        model="claude-sonnet-4-5-20250929",
    )


def test_anthropic_tools():
    """Test Anthropic with tools."""
    llm_client = get_anthropic_client()

    class Math:
        def add(self, a: int, b: int) -> int:
            return a + b + 2

    tools = Tools()
    tools.add_tools(Math())

    runner = AnthropicMessagesRunner(
        tools=tools,
        developer_prompt="use the provided function 'add' when user asks to add numbers",
        llm_client=llm_client
    )

    test_callback = _TestCallback()

    prompt = "how much is 2 + 3"
    result = runner.loop(
        prompt=prompt,
        callback=test_callback,
    )

    # Should have tool calls
    assert len(test_callback.function_calls) == 1
    func_call, _ = test_callback.function_calls[0]
    assert func_call.name == "add"

    # Should have messages and responses
    assert len(test_callback.messages) >= 1
    assert len(test_callback.responses) == 2  # tool call + final response

    # The result should mention 7 (2 + 3 + 2 from the mock function)
    assert "7" in result.last_message


def test_anthropic_simple_message():
    """Test Anthropic with a simple message."""
    llm_client = get_anthropic_client()

    runner = AnthropicMessagesRunner(
        developer_prompt="You are a helpful assistant.",
        llm_client=llm_client
    )

    prompt = "Say 'Hello, test!'"
    result = runner.loop(prompt=prompt)

    assert isinstance(result.last_message, str)
    assert "hello" in result.last_message.lower()
    assert "test" in result.last_message.lower()


def test_anthropic_no_developer_prompt():
    """Test Anthropic with no developer prompt."""
    llm_client = get_anthropic_client()

    runner = AnthropicMessagesRunner(
        llm_client=llm_client
    )

    prompt = "how much is 2 + 3? Reply with just the number."
    result = runner.loop(prompt=prompt)

    # Should contain the answer
    assert "5" in result.last_message


def test_anthropic_conversation():
    """Test Anthropic with conversation history."""
    llm_client = get_anthropic_client()

    runner = AnthropicMessagesRunner(
        developer_prompt="You are a helpful assistant.",
        llm_client=llm_client
    )

    # First message
    result1 = runner.loop("My favorite color is blue.")
    assert "blue" in result1.last_message.lower() or result1.last_message

    # Second message with context
    result2 = runner.loop("What is my favorite color?", previous_messages=result1.all_messages)
    assert "blue" in result2.last_message.lower()
