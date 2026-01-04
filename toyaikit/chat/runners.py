import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.responses.easy_input_message import EasyInputMessage
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel

from toyaikit.chat.interface import ChatInterface
from toyaikit.llm import LLMClient
from toyaikit.pricing import CostInfo, PricingConfig, TokenUsage
from toyaikit.tools import Tools

# T must be either a str or a (subclass)
# instance of pydantic BaseModel
T = TypeVar("T", str, BaseModel)


def _get_tool_call_output(call_result) -> str:
    """Extract output from tool call result, handling both dict and object types."""
    if isinstance(call_result, dict):
        return call_result.get("output", "")
    return getattr(call_result, "output", "")


@dataclass
class LoopResult(Generic[T]):
    new_messages: list
    all_messages: list
    tokens: TokenUsage
    cost: CostInfo
    last_message: T


class RunnerCallback(ABC):
    """Abstract base class for different chat runners."""

    @abstractmethod
    def on_function_call(self, function_call: dict, result: str):
        """
        Called when a function call is made.
        """
        pass

    @abstractmethod
    def on_message(self, message: dict):
        """
        Called when a message is received.
        """
        pass

    @abstractmethod
    def on_reasoning(self, reasoning: str):
        """
        Called when a reasoning is received.
        """
        pass

    @abstractmethod
    def on_response(self, response):
        pass


class ChatRunner(ABC):
    """Abstract base class for different chat runners."""

    def loop(
        self,
        prompt: str,
        previous_messages: list = None,
        callback: RunnerCallback = None,
        output_type: BaseModel = None
    ) -> LoopResult:
        """
        Do one tool-call loop.
        """
        pass

    @abstractmethod
    def run(self, previous_messages: list = None) -> LoopResult:
        """
        Repeast tool call loops until user asks to stop
        """
        pass


class DisplayingRunnerCallback(RunnerCallback):
    def __init__(self, chat_interface: ChatInterface):
        self.chat_interface = chat_interface

    def on_function_call(self, function_call, result):
        self.chat_interface.display_function_call(
            function_call.name,
            function_call.arguments,
            result,
        )

    def on_message(self, message):
        self.chat_interface.display_response(message)

    def on_reasoning(self, reasoning):
        self.chat_interface.display_reasoning(reasoning)

    def on_response(self, response):
        self.chat_interface.display("-> Response received")


class BaseToolUsingRunner(ChatRunner):
    """Base class for runners that use tools and LLM clients.

    Provides common initialization and run() method implementation.
    Subclasses only need to implement the loop() method.
    """

    def __init__(
        self,
        tools: Tools = None,
        developer_prompt: str = "You're a helpful assistant.",
        chat_interface: ChatInterface = None,
        llm_client: LLMClient = None,
    ):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.llm_client = llm_client
        self.displaying_callback = DisplayingRunnerCallback(chat_interface)
        self.pricing_config = PricingConfig()

    @abstractmethod
    def loop(
        self,
        prompt: str,
        previous_messages: list = None,
        callback: RunnerCallback = None,
        output_format: BaseModel = None,
    ) -> LoopResult:
        """Execute one tool-call loop. Must be implemented by subclasses."""
        pass

    def run(
        self,
        previous_messages: list = None,
        stop_criteria: Callable = None,
    ) -> LoopResult:
        """Repeat tool-call loops until user asks to stop."""
        chat_messages = self._initialize_messages(previous_messages)

        total_input_tokens = 0
        total_output_tokens = 0
        last_message_text = ""

        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            loop_result = self.loop(
                prompt=user_input,
                previous_messages=chat_messages,
                callback=self.displaying_callback,
            )

            chat_messages.extend(loop_result.new_messages)
            total_input_tokens += loop_result.tokens.input_tokens
            total_output_tokens += loop_result.tokens.output_tokens
            last_message_text = loop_result.last_message

            if stop_criteria and stop_criteria(loop_result.new_messages):
                break

        combined_cost = self.pricing_config.calculate_cost(
            self.llm_client.model, total_input_tokens, total_output_tokens
        )
        combined_tokens = TokenUsage(
            model=self.llm_client.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        return LoopResult(
            new_messages=chat_messages,
            all_messages=chat_messages,
            tokens=combined_tokens,
            cost=combined_cost,
            last_message=last_message_text,
        )

    @abstractmethod
    def _initialize_messages(self, previous_messages: list = None) -> list:
        """Initialize chat messages. Must be implemented by subclasses."""
        pass


class OpenAIResponsesRunner(BaseToolUsingRunner):
    """Runner for OpenAI responses API."""

    def _initialize_messages(self, previous_messages: list = None) -> list:
        if previous_messages is None or len(previous_messages) == 0:
            return [
                EasyInputMessage(
                    role="developer",
                    content=self.developer_prompt,
                )
            ]
        return list(previous_messages)  # Return a copy

    def loop(
        self,
        prompt: str,
        previous_messages: list[dict] = None,
        callback: RunnerCallback = None,
        output_format: BaseModel = None,
    ) -> LoopResult:
        chat_messages = []
        prev_messages_len = 0

        if previous_messages is None or len(previous_messages) == 0:
            chat_messages.append(
                EasyInputMessage(
                    role="developer",
                    content=self.developer_prompt,
                )
            )
        else:
            chat_messages.extend(previous_messages)
            prev_messages_len = len(previous_messages)

        chat_messages.append(
            EasyInputMessage(
                role="user",
                content=prompt,
            )
        )

        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            response = self.llm_client.send_request(
                chat_messages=chat_messages,
                tools=self.tools,
                output_format=output_format,
            )
            if callback:
                callback.on_response(response)

            if hasattr(response, "usage") and response.usage:
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

            has_function_calls = False

            chat_messages.extend(response.output)

            for entry in response.output:
                if entry.type == "function_call":
                    result = self.tools.function_call(entry)
                    chat_messages.append(result)
                    if callback:
                        callback.on_function_call(entry, result['output'])
                    has_function_calls = True

                elif entry.type == "message":
                    if callback:
                        callback.on_message(entry.content[0].text)

            if not has_function_calls:
                break

        cost_info = self.pricing_config.calculate_cost(
            self.llm_client.model, total_input_tokens, total_output_tokens
        )

        token_usage = TokenUsage(
            model=self.llm_client.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        new_messages = chat_messages[prev_messages_len:]

        last_message_text = ""
        last_message = None
        for entry in reversed(response.output):
            if entry.type == "message":
                last_message_text = entry.content[0].text
                if output_format:
                    last_message = output_format.model_validate_json(last_message_text)
                else:
                    last_message = last_message_text
                break

        return LoopResult(
            new_messages=new_messages,
            all_messages=chat_messages,
            tokens=token_usage,
            cost=cost_info,
            last_message=last_message,
        )


class OpenAIAgentsSDKRunner(ChatRunner):
    """Runner for OpenAI Agents SDK."""

    def __init__(self, chat_interface: ChatInterface, agent):
        try:
            from agents import Runner
        except ImportError:
            raise ImportError(
                "Please run 'pip install openai-agents' to use this feature"
            )

        self.agent = agent
        self.runner = Runner()
        self.chat_interface = chat_interface

    async def run(self) -> None:
        from agents import SQLiteSession

        session_id = f"chat_session_{uuid.uuid4().hex[:8]}"
        session = SQLiteSession(session_id)

        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            result = await self.runner.run(
                self.agent, input=user_input, session=session
            )

            func_calls = {}
            for ni in result.new_items:
                raw = ni.raw_item
                if ni.type == "tool_call_item":
                    func_calls[raw.call_id] = raw

            for ni in result.new_items:
                raw = ni.raw_item

                if ni.type == "handoff_call_item":
                    raw = ni.raw_item
                    self.chat_interface.display(f"handoff: {raw.name}")

                if ni.type == "handoff_output_item":
                    self.chat_interface.display(
                        f"handoff: {ni.target_agent.name} -> {ni.source_agent.name} successful"
                    )

                if ni.type == "tool_call_output_item":
                    call_id = raw["call_id"]
                    if call_id not in func_calls:
                        self.chat_interface.display(
                            f"error: cannot find the call parameters for {call_id=}"
                        )
                    else:
                        func_call = func_calls[call_id]
                        self.chat_interface.display_function_call(
                            func_call.name, func_call.arguments, raw["output"]
                        )

                if ni.type == "message_output_item":
                    md = raw.content[0].text
                    self.chat_interface.display_response(md)


class PydanticAIRunner(ChatRunner):
    """Runner for Pydantic AI."""

    def __init__(self, chat_interface: ChatInterface, agent):
        self.chat_interface = chat_interface
        self.agent = agent

    async def run(self) -> None:
        message_history = []

        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            result = await self.agent.run(
                user_prompt=user_input,
                message_history=message_history
            )

            messages = result.new_messages()
            tool_calls = {}

            for m in messages:
                for part in m.parts:
                    kind = part.part_kind

                    if kind == "text":
                        self.chat_interface.display_response(part.content)

                    elif kind == "tool-call":
                        tool_calls[part.tool_call_id] = part

                    elif kind == "tool-return":
                        call = tool_calls.get(part.tool_call_id)

                        raw_result = part.content
                        if raw_result is None:
                            result_str = ""
                        elif isinstance(raw_result, str):
                            result_str = raw_result
                        else:
                            result_str = json.dumps(raw_result, indent=2, default=str)

                        self.chat_interface.display_function_call(
                            call.tool_name,
                            json.dumps(call.args),
                            result_str
                        )

            message_history.extend(messages)



class OpenAIChatCompletionsRunner(BaseToolUsingRunner):
    """Runner for OpenAI chat completions API."""

    def _initialize_messages(self, previous_messages: list = None) -> list:
        if previous_messages is None or len(previous_messages) == 0:
            return [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self.developer_prompt
                )
            ]
        return list(previous_messages)  # Return a copy

    @staticmethod
    def convert_function_output_to_tool_message(data):
        return ChatCompletionFunctionMessageParam(
            role="tool",
            tool_call_id=data["call_id"],
            content=data["output"],
        )

    def loop(
        self,
        prompt: str,
        previous_messages: list = None,
        callback: RunnerCallback = None,
        output_format: BaseModel = None,
    ) -> LoopResult:
        chat_messages = []
        prev_messages_len = 0

        if previous_messages is None or len(previous_messages) == 0:
            chat_messages.append(
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=self.developer_prompt
                )
            )
        else:
            chat_messages.extend(previous_messages)
            prev_messages_len = len(previous_messages)

        user_message = ChatCompletionUserMessageParam(
            role="user",
            content=prompt
        )
        chat_messages.append(user_message)

        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            reponse = self.llm_client.send_request(
                chat_messages, self.tools, output_format
            )

            if callback:
                callback.on_response(reponse)

            if reponse.usage:
                total_input_tokens += reponse.usage.prompt_tokens
                total_output_tokens += reponse.usage.completion_tokens

            first_choice = reponse.choices[0]
            message_response = first_choice.message
            chat_messages.append(message_response)

            if hasattr(message_response, "reasoning_content"):
                reasoning = (message_response.reasoning_content or "").strip()
                if reasoning != "" and callback:
                    callback.on_reasoning(reasoning)

            content = (message_response.content or "").strip()
            if content != "" and callback:
                callback.on_message(content)

            calls = []

            if hasattr(message_response, "tool_calls"):
                calls = message_response.tool_calls

            if calls is None:
                break

            if len(calls) == 0:
                break

            for call in calls:
                function_call = ResponseFunctionToolCall(
                    type="function_call",
                    name=call.function.name,
                    arguments=call.function.arguments,
                    call_id=call.id,
                )

                call_result = self.tools.function_call(function_call)
                call_result = self.convert_function_output_to_tool_message(call_result)

                chat_messages.append(call_result)

                if callback:
                    content_val = getattr(call_result, "content", None)
                    if content_val is None and isinstance(call_result, dict):
                        content_val = call_result.get("content")
                    callback.on_function_call(function_call, content_val)

        pricing_config = PricingConfig()
        cost = pricing_config.calculate_cost(
            self.llm_client.model, total_input_tokens, total_output_tokens
        )

        token_usage = TokenUsage(
            model=self.llm_client.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        new_messages = chat_messages[prev_messages_len:]

        last_message_text = (message_response.content or "").strip()
        if output_format:
            last_message = output_format.model_validate_json(last_message_text)
        else:
            last_message = last_message_text

        return LoopResult(
            new_messages=new_messages,
            all_messages=chat_messages,
            tokens=token_usage,
            cost=cost,
            last_message=last_message,
        )


class AnthropicMessagesRunner(BaseToolUsingRunner):
    """Runner for Anthropic Messages API."""

    def _initialize_messages(self, previous_messages: list = None) -> list:
        if previous_messages is None or len(previous_messages) == 0:
            return [{
                "role": "system",
                "content": self.developer_prompt
            }]
        return list(previous_messages)  # Return a copy

    def loop(
        self,
        prompt: str,
        previous_messages: list = None,
        callback: RunnerCallback = None,
        output_format: BaseModel = None,
    ) -> LoopResult:
        chat_messages = []
        prev_messages_len = 0

        if previous_messages is None or len(previous_messages) == 0:
            chat_messages.append({
                "role": "system",
                "content": self.developer_prompt
            })
        else:
            chat_messages.extend(previous_messages)
            prev_messages_len = len(previous_messages)

        chat_messages.append({
            "role": "user",
            "content": prompt
        })

        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            response = self.llm_client.send_request(
                chat_messages=chat_messages,
                tools=self.tools,
                output_format=output_format,
            )

            if callback:
                callback.on_response(response)

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

            # Process the response
            assistant_message = {
                "role": "assistant",
                "content": response.content
            }
            chat_messages.append(assistant_message)

            has_tool_calls = False
            text_content = []

            for block in response.content:
                if block.type == "text":
                    text_content.append(block.text)
                    if callback:
                        callback.on_message(block.text)

                elif block.type == "tool_use":
                    has_tool_calls = True
                    function_call = ResponseFunctionToolCall(
                        type="function_call",
                        name=block.name,
                        arguments=json.dumps(block.input),
                        call_id=block.id,
                    )

                    call_result = self.tools.function_call(function_call)
                    result_output = _get_tool_call_output(call_result)

                    # Anthropic expects tool results in a user message with tool_result blocks
                    tool_result_message = {
                        "role": "tool",
                        "tool_call_id": block.id,
                        "content": result_output
                    }
                    chat_messages.append(tool_result_message)

                    if callback:
                        callback.on_function_call(function_call, result_output)

            if not has_tool_calls:
                break

        cost_info = self.pricing_config.calculate_cost(
            self.llm_client.model, total_input_tokens, total_output_tokens
        )

        token_usage = TokenUsage(
            model=self.llm_client.model,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        new_messages = chat_messages[prev_messages_len:]

        # Extract last message text
        last_message_text = ""
        last_message = None
        if text_content:
            last_message_text = "".join(text_content)
            if output_format:
                last_message = output_format.model_validate_json(last_message_text)
            else:
                last_message = last_message_text

        return LoopResult(
            new_messages=new_messages,
            all_messages=chat_messages,
            tokens=token_usage,
            cost=cost_info,
            last_message=last_message,
        )
