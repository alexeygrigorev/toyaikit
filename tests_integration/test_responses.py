from pydantic import BaseModel

from tests_integration.utils import find_function_calls_responses
from toyaikit.chat.runners import OpenAIResponsesRunner
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools

from .utils import _TestCallback


def test_responses_api_tools_structured_output():
    llm_client = OpenAIClient(model='gpt-4o-mini')

    class Math:
        def add(self, a: int, b: int) -> int:
            return a + b + 2

    tools = Tools()
    tools.add_tools(Math())

    class Result(BaseModel):
        result: int

    runner = OpenAIResponsesRunner(
        tools=tools,
        developer_prompt="use the provided function 'add' when user asks to add numbers",
        llm_client=llm_client
    )

    prompt = "how much is 2 + 3"

    test_callback = _TestCallback()

    result = runner.loop(
        prompt=prompt,
        callback=test_callback,
        output_format=Result,
    )
    
    messages = result.all_messages
    calls = find_function_calls_responses(messages)
    assert any(name == "add" for name, _ in calls), "Expected 'add' tool call to occur"

    assert len(test_callback.messages) == 1  
    assert len(test_callback.function_calls) == 1
    assert len(test_callback.responses) == 2

    output = result.last_message
    assert output.result == (2 + 3 + 2)


def test_responses_api_structured_output():
    llm_client = OpenAIClient(model='gpt-4o-mini')

    class Result(BaseModel):
        result: int

    runner = OpenAIResponsesRunner(
        developer_prompt="help user with arithmetics",
        llm_client=llm_client
    )

    prompt = "how much is 2 + 3"
    result = runner.loop(prompt=prompt, output_format=Result)
    # no tool call assert here (no tools provided)

    output = result.last_message
    assert output.result == (2 + 3)


def test_responses_api_no_developer_prompt():
    llm_client = OpenAIClient(model='gpt-4o-mini')

    class Result(BaseModel):
        result: int

    runner = OpenAIResponsesRunner(
        llm_client=llm_client
    )

    prompt = "how much is 2 + 3"
    result = runner.loop(prompt=prompt, output_format=Result)
    # no tool call assert here (no tools provided)

    output = result.last_message
    assert output.result == (2 + 3)
