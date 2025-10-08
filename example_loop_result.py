"""
Example of using the LoopResult return type from runners.

The loop() method now returns a LoopResult object with:
- new_messages: Messages added in this loop iteration
- all_messages: Complete message history
- tokens: TokenUsage with input/output/total token counts
- cost: CostInfo with input/output/total costs in USD
"""

from toyaikit.chat.runners import (
    OpenAIResponsesRunner,
    OpenAIChatCompletionsRunner,
    LoopResult,
)
from toyaikit.llm import OpenAIClient, OpenAIChatCompletionsClient
from toyaikit.chat.interface import ChatInterface


class MockInterface(ChatInterface):
    def input(self):
        return "test"

    def display(self, message):
        print(message)

    def display_response(self, message):
        print(f"Response: {message}")

    def display_function_call(self, name, arguments, result):
        print(f"Function call: {name}({arguments}) -> {result}")

    def display_reasoning(self, reasoning):
        print(f"Reasoning: {reasoning}")


def example_with_responses_api():
    """Example using OpenAI Responses API"""
    runner = OpenAIResponsesRunner(
        developer_prompt="You're a helpful assistant.",
        chat_interface=MockInterface(),
        llm_client=OpenAIClient(model="gpt-4o-mini"),
    )

    result = runner.loop("Tell me a joke")

    assert isinstance(result, LoopResult)

    print(f"\n=== Loop Result ===")
    print(f"New messages count: {len(result.new_messages)}")
    print(f"Total messages count: {len(result.all_messages)}")
    print(f"\n=== Token Usage ===")
    print(f"Input tokens: {result.tokens.input_tokens}")
    print(f"Output tokens: {result.tokens.output_tokens}")
    print(f"Total tokens: {result.tokens.total_tokens}")
    print(f"\n=== Cost ===")
    print(f"Input cost: ${result.cost.input_cost:.6f}")
    print(f"Output cost: ${result.cost.output_cost:.6f}")
    print(f"Total cost: ${result.cost.total_cost:.6f}")


def example_with_chat_completions():
    """Example using OpenAI Chat Completions API"""
    runner = OpenAIChatCompletionsRunner(
        tools=None,
        developer_prompt="You're a helpful assistant.",
        chat_interface=MockInterface(),
        llm_client=OpenAIChatCompletionsClient(model="gpt-4o-mini"),
    )

    result = runner.loop("What's 2+2?")

    assert isinstance(result, LoopResult)

    print(f"\n=== Loop Result ===")
    print(f"New messages count: {len(result.new_messages)}")
    print(f"All messages count: {len(result.all_messages)}")
    print(f"\n=== Token Usage ===")
    print(f"Input tokens: {result.tokens.input_tokens}")
    print(f"Output tokens: {result.tokens.output_tokens}")
    print(f"Total tokens: {result.tokens.total_tokens}")
    print(f"\n=== Cost ===")
    print(f"Input cost: ${result.cost.input_cost:.6f}")
    print(f"Output cost: ${result.cost.output_cost:.6f}")
    print(f"Total cost: ${result.cost.total_cost:.6f}")


if __name__ == "__main__":
    print("=" * 50)
    print("Example 1: OpenAI Responses API")
    print("=" * 50)
    # example_with_responses_api()  # Uncomment to run

    print("\n" + "=" * 50)
    print("Example 2: OpenAI Chat Completions API")
    print("=" * 50)
    # example_with_chat_completions()  # Uncomment to run

    print("\nExamples are commented out to avoid API calls.")
    print("Uncomment the function calls to test with real API.")
