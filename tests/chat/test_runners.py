import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from toyaikit.chat.interface import ChatInterface
from toyaikit.chat.runners import (
    ChatRunner,
    DisplayingRunnerCallback,
    LoopResult,
    OpenAIAgentsSDKRunner,
    OpenAIChatCompletionsRunner,
    OpenAIResponsesRunner,
    PydanticAIRunner,
    RunnerCallback,
)
from toyaikit.llm import LLMClient
from toyaikit.pricing import CostInfo, TokenUsage
from toyaikit.tools import Tools


class TestRunnerCallback:
    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that RunnerCallback cannot be instantiated directly"""
        with pytest.raises(TypeError):
            RunnerCallback()

    def test_concrete_class_must_implement_all_methods(self):
        """Test that concrete class must implement all abstract methods"""

        class IncompleteCallback(RunnerCallback):
            def on_function_call(self, function_call: dict, result: str):
                pass

            # Missing other methods

        with pytest.raises(TypeError):
            IncompleteCallback()

    def test_complete_concrete_implementation(self):
        """Test that complete implementation works"""

        class CompleteCallback(RunnerCallback):
            def on_function_call(self, function_call: dict, result: str):
                pass

            def on_message(self, message: dict):
                pass

            def on_reasoning(self, reasoning: str):
                pass

            def on_response(self, response):
                pass

        # Should not raise
        callback = CompleteCallback()
        assert isinstance(callback, RunnerCallback)


class TestChatRunner:
    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that ChatRunner cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ChatRunner()

    def test_concrete_class_must_implement_run_method(self):
        """Test that concrete class must implement run method"""

        class IncompleteRunner(ChatRunner):
            # Missing run method implementation
            pass

        with pytest.raises(TypeError):
            IncompleteRunner()

    def test_complete_concrete_implementation(self):
        """Test that complete implementation works"""

        class CompleteRunner(ChatRunner):
            def run(self, previous_messages: list = None) -> list:
                return []

        # Should not raise
        runner = CompleteRunner()
        assert isinstance(runner, ChatRunner)

        # Test that loop method returns None (not abstract)
        result = runner.loop("test prompt")
        assert result is None


class TestDisplayingRunnerCallback:
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_interface = Mock(spec=ChatInterface)
        self.callback = DisplayingRunnerCallback(self.mock_interface)

    def test_initialization(self):
        """Test DisplayingRunnerCallback initialization"""
        assert self.callback.chat_interface == self.mock_interface

    def test_on_function_call(self):
        """Test on_function_call delegates to chat interface"""
        function_call = SimpleNamespace(
            name="test_func", arguments='{"param": "value"}'
        )
        result = "test result"

        self.callback.on_function_call(function_call, result)

        self.mock_interface.display_function_call.assert_called_once_with(
            "test_func", '{"param": "value"}', result
        )

    def test_on_message(self):
        """Test on_message delegates to chat interface"""
        message = "Test message"

        self.callback.on_message(message)

        self.mock_interface.display_response.assert_called_once_with(message)

    def test_on_reasoning(self):
        """Test on_reasoning delegates to chat interface"""
        reasoning = "Test reasoning"

        self.callback.on_reasoning(reasoning)

        self.mock_interface.display_reasoning.assert_called_once_with(reasoning)

    def test_on_response(self):
        """Test on_response creates log message and displays it"""
        response = SimpleNamespace(output=["item1", "item2", "item3"])

        self.callback.on_response(response)

        expected_log = f"response with 3, {response}"
        self.mock_interface.display.assert_called_once_with(expected_log)


class TestOpenAIResponsesRunner:
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_tools = Mock(spec=Tools)
        self.mock_interface = Mock(spec=ChatInterface)
        self.mock_llm_client = Mock(spec=LLMClient)

        self.runner = OpenAIResponsesRunner(
            tools=self.mock_tools,
            developer_prompt="Test developer prompt",
            chat_interface=self.mock_interface,
            llm_client=self.mock_llm_client,
        )

    def test_initialization(self):
        """Test OpenAIResponsesRunner initialization"""
        assert self.runner.tools == self.mock_tools
        assert self.runner.developer_prompt == "Test developer prompt"
        assert self.runner.chat_interface == self.mock_interface
        assert self.runner.llm_client == self.mock_llm_client
        assert isinstance(self.runner.displaying_callback, DisplayingRunnerCallback)

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters"""
        runner = OpenAIResponsesRunner()

        assert runner.tools is None
        assert runner.developer_prompt == "You're a helpful assistant."
        assert runner.chat_interface is None
        assert runner.llm_client is None

    def test_loop_with_no_previous_messages(self):
        """Test loop method with no previous messages"""
        # Mock response with message (no function calls)
        message_entry = SimpleNamespace(
            type="message", content=[SimpleNamespace(text="Hello")]
        )
        mock_usage = SimpleNamespace(model="gpt-4o-mini", input_tokens=10, output_tokens=20)
        mock_response = SimpleNamespace(output=[message_entry], usage=mock_usage)
        self.mock_llm_client.send_request.return_value = mock_response
        self.mock_llm_client.model = "gpt-4o-mini"

        result = self.runner.loop("Test prompt")

        # Should add developer prompt and user message
        expected_messages = [
            {"role": "developer", "content": "Test developer prompt"},
            {"role": "user", "content": "Test prompt"},
            message_entry,
        ]

        self.mock_llm_client.send_request.assert_called_once_with(
            chat_messages=expected_messages, tools=self.mock_tools
        )

        # Should return LoopResult
        assert isinstance(result, LoopResult)
        assert len(result.new_messages) == 3
        assert result.new_messages[0] == {
            "role": "developer",
            "content": "Test developer prompt",
        }
        assert result.new_messages[1] == {"role": "user", "content": "Test prompt"}
        assert result.new_messages[2] == message_entry
        assert result.tokens.model == "gpt-4o-mini"
        assert result.tokens.input_tokens == 10
        assert result.tokens.output_tokens == 20
        assert isinstance(result.cost, CostInfo)
        assert result.cost.total_cost > 0
        assert result.last_message == "Hello"

    def test_loop_with_previous_messages(self):
        """Test loop method with previous messages"""
        previous_messages = [{"role": "system", "content": "Previous message"}]

        message_entry = SimpleNamespace(
            type="message", content=[SimpleNamespace(text="Hello")]
        )
        mock_usage = SimpleNamespace(input_tokens=10, output_tokens=20)
        mock_response = SimpleNamespace(output=[message_entry], usage=mock_usage)
        self.mock_llm_client.send_request.return_value = mock_response
        self.mock_llm_client.model = "gpt-4o-mini"

        result = self.runner.loop("Test prompt", previous_messages=previous_messages)

        expected_messages = [
            {"role": "system", "content": "Previous message"},
            {"role": "user", "content": "Test prompt"},
            message_entry,
        ]

        self.mock_llm_client.send_request.assert_called_once_with(
            chat_messages=expected_messages, tools=self.mock_tools
        )

        # Should return only the new messages (after previous_messages_len)
        assert isinstance(result, LoopResult)
        assert result.new_messages == [
            {"role": "user", "content": "Test prompt"},
            message_entry,
        ]

    def test_loop_with_function_calls(self):
        """Test loop method with function calls"""
        # Mock function call entry
        function_call = SimpleNamespace(
            type="function_call", name="test_func", arguments="{}"
        )
        function_result = {"role": "function", "content": "Function result"}
        self.mock_tools.function_call.return_value = function_result

        # Mock final message after function call
        message_entry = SimpleNamespace(
            type="message", content=[SimpleNamespace(text="Final response")]
        )

        mock_usage = SimpleNamespace(input_tokens=10, output_tokens=20)
        # First response has function call, second has final message
        self.mock_llm_client.send_request.side_effect = [
            SimpleNamespace(output=[function_call], usage=mock_usage),
            SimpleNamespace(output=[message_entry], usage=mock_usage),
        ]
        self.mock_llm_client.model = "gpt-4o-mini"

        mock_callback = Mock(spec=RunnerCallback)
        self.runner.loop("Test prompt", callback=mock_callback)

        # Should call function_call on tools
        self.mock_tools.function_call.assert_called_once_with(function_call)

        # Should call callback
        mock_callback.on_function_call.assert_called_once_with(
            function_call, function_result
        )

        # Should make two LLM calls
        assert self.mock_llm_client.send_request.call_count == 2

    def test_loop_with_message_callback(self):
        """Test loop method calls message callback"""
        message_entry = SimpleNamespace(
            type="message", content=[SimpleNamespace(text="Hello")]
        )
        mock_usage = SimpleNamespace(input_tokens=10, output_tokens=20)
        mock_response = SimpleNamespace(output=[message_entry], usage=mock_usage)
        self.mock_llm_client.send_request.return_value = mock_response
        self.mock_llm_client.model = "gpt-4o-mini"

        mock_callback = Mock(spec=RunnerCallback)
        self.runner.loop("Test prompt", callback=mock_callback)

        mock_callback.on_message.assert_called_once_with("Hello")

    def test_run_interactive_chat(self):
        """Test run method for interactive chat"""
        # Mock user inputs: first "hello", then "stop"
        self.mock_interface.input.side_effect = ["hello", "stop"]
        self.mock_llm_client.model = "gpt-4o-mini"

        # Mock loop response
        with patch.object(self.runner, "loop") as mock_loop:
            mock_loop.return_value = LoopResult(
                new_messages=[{"role": "assistant", "content": "Hi there"}],
                all_messages=[{"role": "assistant", "content": "Hi there"}],
                tokens=TokenUsage(model="gpt-4o", input_tokens=10, output_tokens=20),
                cost=CostInfo(input_cost=0.001, output_cost=0.002, total_cost=0.003),
                last_message="Hi there",
            )

            self.runner.run()

            # Should call input twice
            assert self.mock_interface.input.call_count == 2

            # Should display "Chat ended."
            self.mock_interface.display.assert_called_with("Chat ended.")

            # Should call loop once
            mock_loop.assert_called_once()

    def test_run_with_previous_messages(self):
        """Test run method with previous messages"""
        previous_messages = [{"role": "system", "content": "Previous"}]
        self.mock_interface.input.side_effect = ["hello", "stop"]
        self.mock_llm_client.model = "gpt-4o-mini"

        with patch.object(self.runner, "loop") as mock_loop:
            mock_loop.return_value = LoopResult(
                new_messages=[{"role": "assistant", "content": "Hi"}],
                all_messages=[{"role": "assistant", "content": "Hi"}],
                tokens=TokenUsage(model="gpt-4o", input_tokens=10, output_tokens=20),
                cost=CostInfo(input_cost=0.001, output_cost=0.002, total_cost=0.003),
                last_message="Hi",
            )

            self.runner.run(previous_messages=previous_messages)

            # Should only call loop once since second input is "stop"
            assert mock_loop.call_count == 1

            # Check the call to loop - should get the original previous_messages
            call_args = mock_loop.call_args_list[0]
            assert call_args[1]["prompt"] == "hello"
            # The previous_messages should start with our input but may be accumulated
            actual_previous = call_args[1]["previous_messages"]
            assert actual_previous[0] == {
                "role": "system",
                "content": "Previous",
            }
            assert call_args[1]["callback"] == self.runner.displaying_callback

    def test_run_with_stop_criteria(self):
        """Test run method with stop criteria function"""
        self.mock_interface.input.side_effect = ["hello", "world", "more"]
        self.mock_llm_client.model = "gpt-4o-mini"

        # Stop criteria that returns True after second message
        def stop_criteria(messages):
            return len(messages) > 0 and "stop" in str(messages)

        with patch.object(self.runner, "loop") as mock_loop:
            mock_loop.side_effect = [
                LoopResult(
                    new_messages=[{"role": "assistant", "content": "Hi"}],
                    all_messages=[{"role": "assistant", "content": "Hi"}],
                    tokens=TokenUsage(model="gpt-4o", input_tokens=10, output_tokens=20),
                    cost=CostInfo(input_cost=0.001, output_cost=0.002, total_cost=0.003),
                    last_message="Hi",
                ),
                LoopResult(
                    new_messages=[{"role": "assistant", "content": "Please stop"}],
                    all_messages=[{"role": "assistant", "content": "Please stop"}],
                    tokens=TokenUsage(model="gpt-4o", input_tokens=10, output_tokens=20),
                    cost=CostInfo(input_cost=0.001, output_cost=0.002, total_cost=0.003),
                    last_message="Please stop",
                ),
            ]

            self.runner.run(stop_criteria=stop_criteria)

            # Should stop after second loop call due to stop criteria
            assert mock_loop.call_count == 2


class TestOpenAIAgentsSDKRunner:
    def test_initialization_with_agents_import_error(self):
        """Test initialization fails gracefully when agents SDK not available"""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'agents'"),
        ):
            with pytest.raises(
                ImportError,
                match="Please run 'pip install openai-agents' to use this feature",
            ):
                OpenAIAgentsSDKRunner(Mock(), Mock())

    def test_initialization_success(self):
        """Test successful initialization when agents SDK is available"""
        mock_interface = Mock()
        mock_agent = Mock()

        runner = OpenAIAgentsSDKRunner(mock_interface, mock_agent)

        assert runner.agent == mock_agent
        assert runner.chat_interface == mock_interface
        # runner.runner should be set to Runner() instance

    @pytest.mark.asyncio
    async def test_run_interactive_chat(self):
        """Test async run method"""
        mock_interface = Mock()
        mock_agent = Mock()

        # Mock user inputs
        mock_interface.input.side_effect = ["hello", "stop"]

        runner = OpenAIAgentsSDKRunner(mock_interface, mock_agent)

        # Mock the runner.run method
        mock_result = Mock()
        mock_result.new_items = []

        with patch.object(runner.runner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await runner.run()

            # Should call input twice
            assert mock_interface.input.call_count == 2

            # Should display "Chat ended."
            mock_interface.display.assert_called_with("Chat ended.")

    @pytest.mark.asyncio
    async def test_run_with_tool_calls(self):
        """Test run method handles tool calls"""
        mock_interface = Mock()
        mock_agent = Mock()

        mock_interface.input.side_effect = ["test", "stop"]

        runner = OpenAIAgentsSDKRunner(mock_interface, mock_agent)

        # Mock result with tool calls
        mock_tool_call = Mock()
        mock_tool_call.call_id = "call_123"
        mock_tool_call.name = "test_tool"
        mock_tool_call.arguments = '{"param": "value"}'

        mock_tool_output = {"call_id": "call_123", "output": "tool result"}

        mock_item1 = Mock()
        mock_item1.type = "tool_call_item"
        mock_item1.raw_item = mock_tool_call

        mock_item2 = Mock()
        mock_item2.type = "tool_call_output_item"
        mock_item2.raw_item = mock_tool_output

        mock_result = Mock()
        mock_result.new_items = [mock_item1, mock_item2]

        with patch.object(runner.runner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await runner.run()

            # Should display function call
            mock_interface.display_function_call.assert_called_once_with(
                "test_tool", '{"param": "value"}', "tool result"
            )

    @pytest.mark.asyncio
    async def test_run_with_handoff_calls(self):
        """Test run method handles handoff calls"""
        mock_interface = Mock()
        mock_agent = Mock()

        mock_interface.input.side_effect = ["test", "stop"]

        runner = OpenAIAgentsSDKRunner(mock_interface, mock_agent)

        # Mock handoff call item
        mock_handoff_call = Mock()
        mock_handoff_call.name = "agent_handoff"

        mock_item1 = Mock()
        mock_item1.type = "handoff_call_item"
        mock_item1.raw_item = mock_handoff_call

        mock_result = Mock()
        mock_result.new_items = [mock_item1]

        with patch.object(runner.runner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await runner.run()

            # Should display handoff call - check all calls to display
            display_calls = [
                call[0][0] for call in mock_interface.display.call_args_list
            ]
            assert "handoff: agent_handoff" in display_calls

    @pytest.mark.asyncio
    async def test_run_with_handoff_output(self):
        """Test run method handles handoff output"""
        mock_interface = Mock()
        mock_agent = Mock()

        mock_interface.input.side_effect = ["test", "stop"]

        runner = OpenAIAgentsSDKRunner(mock_interface, mock_agent)

        # Mock handoff output item
        mock_target_agent = Mock()
        mock_target_agent.name = "target_agent"
        mock_source_agent = Mock()
        mock_source_agent.name = "source_agent"

        mock_item = Mock()
        mock_item.type = "handoff_output_item"
        mock_item.target_agent = mock_target_agent
        mock_item.source_agent = mock_source_agent

        mock_result = Mock()
        mock_result.new_items = [mock_item]

        with patch.object(runner.runner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await runner.run()

            # Should display handoff success - check all calls to display
            display_calls = [
                call[0][0] for call in mock_interface.display.call_args_list
            ]
            assert "handoff: target_agent -> source_agent successful" in display_calls

    @pytest.mark.asyncio
    async def test_run_with_message_output(self):
        """Test run method handles message output"""
        mock_interface = Mock()
        mock_agent = Mock()

        mock_interface.input.side_effect = ["test", "stop"]

        runner = OpenAIAgentsSDKRunner(mock_interface, mock_agent)

        # Mock message output item
        mock_content = Mock()
        mock_content.text = "Hello from agent"

        mock_raw = Mock()
        mock_raw.content = [mock_content]

        mock_item = Mock()
        mock_item.type = "message_output_item"
        mock_item.raw_item = mock_raw

        mock_result = Mock()
        mock_result.new_items = [mock_item]

        with patch.object(runner.runner, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result

            await runner.run()

            # Should display response
            mock_interface.display_response.assert_called_once_with("Hello from agent")


class TestPydanticAIRunner:
    def test_initialization(self):
        """Test PydanticAIRunner initialization"""
        mock_interface = Mock()
        mock_agent = Mock()

        runner = PydanticAIRunner(mock_interface, mock_agent)

        assert runner.chat_interface == mock_interface
        assert runner.agent == mock_agent

    @pytest.mark.asyncio
    async def test_run_interactive_chat(self):
        """Test async run method"""
        mock_interface = Mock()
        mock_agent = Mock()

        mock_interface.input.side_effect = ["hello", "stop"]

        runner = PydanticAIRunner(mock_interface, mock_agent)

        # Mock agent response
        mock_result = Mock()
        mock_result.new_messages.return_value = []

        mock_agent.run = AsyncMock(return_value=mock_result)

        await runner.run()

        # Should call input twice
        assert mock_interface.input.call_count == 2

        # Should display "Chat ended."
        mock_interface.display.assert_called_with("Chat ended.")

    @pytest.mark.asyncio
    async def test_run_with_tool_calls(self):
        """Test run method handles tool calls and responses"""
        mock_interface = Mock()
        mock_agent = Mock()

        mock_interface.input.side_effect = ["test", "stop"]

        runner = PydanticAIRunner(mock_interface, mock_agent)

        # Mock message parts
        text_part = Mock()
        text_part.part_kind = "text"
        text_part.content = "Hello response"

        tool_call_part = Mock()
        tool_call_part.part_kind = "tool-call"
        tool_call_part.tool_call_id = "call_123"
        tool_call_part.tool_name = "test_tool"
        tool_call_part.args = {"param": "value"}

        tool_return_part = Mock()
        tool_return_part.part_kind = "tool-return"
        tool_return_part.tool_call_id = "call_123"
        tool_return_part.content = "tool result"

        # Mock message with parts
        mock_message = Mock()
        mock_message.parts = [text_part, tool_call_part, tool_return_part]

        mock_result = Mock()
        mock_result.new_messages.return_value = [mock_message]

        mock_agent.run = AsyncMock(return_value=mock_result)

        await runner.run()

        # Should display text response
        mock_interface.display_response.assert_called_with("Hello response")

        # Should display function call
        mock_interface.display_function_call.assert_called_once_with(
            "test_tool", json.dumps({"param": "value"}), "tool result"
        )


class TestOpenAIChatCompletionsRunner:
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_tools = Mock(spec=Tools)
        self.mock_interface = Mock(spec=ChatInterface)
        self.mock_llm_client = Mock(spec=LLMClient)

        self.runner = OpenAIChatCompletionsRunner(
            tools=self.mock_tools,
            developer_prompt="Test system prompt",
            chat_interface=self.mock_interface,
            llm_client=self.mock_llm_client,
        )

    def test_initialization(self):
        """Test OpenAIChatCompletionsRunner initialization"""
        assert self.runner.tools == self.mock_tools
        assert self.runner.developer_prompt == "Test system prompt"
        assert self.runner.chat_interface == self.mock_interface
        assert self.runner.llm_client == self.mock_llm_client

    def test_convert_function_output_to_tool_message(self):
        """Test convert_function_output_to_tool_message method"""
        data = {"call_id": "call_123", "output": "Function result"}

        result = self.runner.convert_function_output_to_tool_message(data)

        expected = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Function result",
        }

        assert result == expected

    def test_loop_simple_message(self):
        """Test loop method with simple message response"""
        # Mock response message
        mock_message = Mock()
        mock_message.content = "Hello there"
        mock_message.tool_calls = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        self.mock_llm_client.send_request.return_value = mock_response
        self.mock_llm_client.model = "gpt-4o-mini"

        mock_callback = Mock()
        result = self.runner.loop("Test prompt", callback=mock_callback)

        # Should call message callback
        mock_callback.on_message.assert_called_once_with("Hello there")

        # Should return LoopResult
        assert isinstance(result, LoopResult)
        assert len(result.new_messages) >= 2

    def test_loop_with_reasoning(self):
        """Test loop method with reasoning content"""
        # Mock response with reasoning
        mock_message = Mock()
        mock_message.content = "Hello"
        mock_message.reasoning_content = "This is my reasoning"
        mock_message.tool_calls = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        self.mock_llm_client.send_request.return_value = mock_response
        self.mock_llm_client.model = "gpt-4o-mini"

        mock_callback = Mock()
        self.runner.loop("Test prompt", callback=mock_callback)

        # Should call reasoning callback
        mock_callback.on_reasoning.assert_called_once_with("This is my reasoning")

    def test_loop_with_tool_calls(self):
        """Test loop method with tool calls"""
        # Mock tool call
        mock_function = Mock()
        mock_function.model_dump.return_value = {
            "name": "test_func",
            "arguments": '{"param": "value"}',
        }

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function

        # Mock response with tool calls
        mock_message = Mock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message

        # Mock final response after tool call
        mock_final_message = Mock()
        mock_final_message.content = "Final response"
        mock_final_message.tool_calls = None

        mock_final_choice = Mock()
        mock_final_choice.message = mock_final_message

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        self.mock_llm_client.send_request.side_effect = [
            Mock(choices=[mock_choice], usage=mock_usage),
            Mock(choices=[mock_final_choice], usage=mock_usage),
        ]
        self.mock_llm_client.model = "gpt-4o-mini"

        # Mock tool function call result
        tool_result = {"call_id": "call_123", "output": "Tool output"}
        self.mock_tools.function_call.return_value = tool_result

        mock_callback = Mock()
        self.runner.loop("Test prompt", callback=mock_callback)

        # Should call tools.function_call
        expected_function_call = {
            "name": "test_func",
            "arguments": '{"param": "value"}',
            "call_id": "call_123",
        }
        self.mock_tools.function_call.assert_called_once_with(expected_function_call)

        # Should call function call callback
        mock_callback.on_function_call.assert_called_once_with(
            expected_function_call, "Tool output"
        )

    def test_run_interactive_chat(self):
        """Test run method for interactive chat"""
        self.mock_interface.input.side_effect = ["hello", "stop"]
        self.mock_llm_client.model = "gpt-4o-mini"

        with patch.object(self.runner, "loop") as mock_loop:
            mock_loop.return_value = LoopResult(
                new_messages=[{"role": "assistant", "content": "Hi"}],
                all_messages=[{"role": "assistant", "content": "Hi"}],
                tokens=TokenUsage(model="gpt-4o", input_tokens=10, output_tokens=20),
                cost=CostInfo(input_cost=0.001, output_cost=0.002, total_cost=0.003),
                last_message="Hi",
            )

            self.runner.run()

            # Should call input twice
            assert self.mock_interface.input.call_count == 2

            # Should display "Chat ended"
            self.mock_interface.display.assert_called_with("Chat ended")

    def test_loop_with_previous_messages_empty_case(self):
        """Test loop method with empty previous messages list"""
        previous_messages = []

        mock_message = Mock()
        mock_message.content = "Hello"
        mock_message.tool_calls = None

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        self.mock_llm_client.send_request.return_value = mock_response
        self.mock_llm_client.model = "gpt-4o-mini"

        result = self.runner.loop("Test prompt", previous_messages=previous_messages)

        # Should treat empty list same as None - add system prompt
        assert isinstance(result, LoopResult)
        assert len(result.new_messages) >= 2

    def test_loop_with_empty_tool_calls_list(self):
        """Test loop method when tool_calls is empty list"""
        # Mock response with empty tool calls list
        mock_message = Mock()
        mock_message.content = "Hello"
        mock_message.tool_calls = []  # Empty list should break loop

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        self.mock_llm_client.send_request.return_value = mock_response
        self.mock_llm_client.model = "gpt-4o-mini"

        mock_callback = Mock()
        self.runner.loop("Test prompt", callback=mock_callback)

        # Should call message callback
        mock_callback.on_message.assert_called_once_with("Hello")

        # Should make only one LLM call since empty tool_calls breaks the loop
        assert self.mock_llm_client.send_request.call_count == 1

    def test_run_with_stop_criteria(self):
        """Test run method with stop criteria"""
        self.mock_interface.input.side_effect = ["hello", "world"]
        self.mock_llm_client.model = "gpt-4o-mini"

        def stop_criteria(messages):
            return any("stop" in str(msg) for msg in messages)

        with patch.object(self.runner, "loop") as mock_loop:
            mock_loop.side_effect = [
                LoopResult(
                    new_messages=[{"role": "assistant", "content": "Hi"}],
                    all_messages=[{"role": "assistant", "content": "Hi"}],
                    tokens=TokenUsage(model="gpt-4o", input_tokens=10, output_tokens=20),
                    cost=CostInfo(input_cost=0.001, output_cost=0.002, total_cost=0.003),
                    last_message="Hi",
                ),
                LoopResult(
                    new_messages=[{"role": "assistant", "content": "Please stop now"}],
                    all_messages=[{"role": "assistant", "content": "Please stop now"}],
                    tokens=TokenUsage(model="gpt-4o", input_tokens=10, output_tokens=20),
                    cost=CostInfo(input_cost=0.001, output_cost=0.002, total_cost=0.003),
                    last_message="Please stop now",
                ),
            ]

            self.runner.run(stop_criteria=stop_criteria)

            # Should stop after second message due to stop criteria
            assert mock_loop.call_count == 2


class TestAbstractMethodCoverage:
    """Test coverage of abstract method pass statements and default implementations"""

    def test_chat_runner_loop_returns_none(self):
        """Test that ChatRunner.loop returns None by default"""

        class TestChatRunner(ChatRunner):
            def run(self, previous_messages: list = None) -> list:
                return []

        runner = TestChatRunner()
        result = runner.loop("test prompt")
        assert result is None

        # Test with all parameters
        result = runner.loop("test", ["prev"], Mock())
        assert result is None
