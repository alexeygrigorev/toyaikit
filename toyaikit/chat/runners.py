import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from toyaikit.chat.interface import ChatInterface
from toyaikit.llm import LLMClient
from toyaikit.pricing import CostInfo, PricingConfig, TokenUsage
from toyaikit.tools import Tools


@dataclass
class LoopResult:
    new_messages: list
    all_messages: list
    tokens: TokenUsage
    cost: CostInfo
    last_message: str


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
    ) -> LoopResult:
        """
        Loop the chat.
        """
        pass

    @abstractmethod
    def run(self, previous_messages: list = None) -> list:
        """
        Run the chat.
        """
        pass


class DisplayingRunnerCallback(RunnerCallback):
    def __init__(self, chat_interface: ChatInterface):
        self.chat_interface = chat_interface

    def on_function_call(self, function_call, result):
        self.chat_interface.display_function_call(
            function_call.name, function_call.arguments, result
        )

    def on_message(self, message):
        self.chat_interface.display_response(message)

    def on_reasoning(self, reasoning):
        self.chat_interface.display_reasoning(reasoning)

    def on_response(self, response):
        log = f"response with {len(response.output)}, {response}"
        self.chat_interface.display(log)


class OpenAIResponsesRunner(ChatRunner):
    """Runner for OpenAI responses API."""

    def __init__(
        self,
        tools: Tools = None,
        developer_prompt: str = "You're a helpful assistant.",
        chat_interface: ChatInterface = None,
        llm_client: LLMClient = None,
    ):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.llm_client = llm_client
        self.chat_interface = chat_interface
        self.displaying_callback = DisplayingRunnerCallback(chat_interface)
        self.pricing_config = PricingConfig()

    def loop(
        self,
        prompt: str,
        previous_messages: list[dict] = None,
        callback: RunnerCallback = None,
    ) -> LoopResult:
        chat_messages = []
        prev_messages_len = 0

        if previous_messages is None or len(previous_messages) == 0:
            chat_messages.append(
                {"role": "developer", "content": self.developer_prompt}
            )
        else:
            chat_messages.extend(previous_messages)
            prev_messages_len = len(previous_messages)

        chat_messages.append({"role": "user", "content": prompt})

        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            response = self.llm_client.send_request(
                chat_messages=chat_messages,
                tools=self.tools,
            )

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
                        callback.on_function_call(entry, result)
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
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )

        new_messages = chat_messages[prev_messages_len:]

        last_message_text = ""
        for entry in reversed(response.output):
            if entry.type == "message":
                last_message_text = entry.content[0].text
                break

        return LoopResult(
            new_messages=new_messages,
            all_messages=chat_messages,
            tokens=token_usage,
            cost=cost_info,
            last_message=last_message_text,
        )

    def run(
        self,
        previous_messages: list = None,
        stop_criteria: Callable = None,
    ) -> LoopResult:
        if previous_messages is None or len(previous_messages) == 0:
            chat_messages = [
                {"role": "developer", "content": self.developer_prompt},
            ]
        else:
            chat_messages = []
            chat_messages.extend(previous_messages)

        total_input_tokens = 0
        total_output_tokens = 0
        last_message_text = ""

        while True:
            question = self.chat_interface.input()
            if question.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            loop_result = self.loop(
                prompt=question,
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
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )

        return LoopResult(
            new_messages=chat_messages,
            all_messages=chat_messages,
            tokens=combined_tokens,
            cost=combined_cost,
            last_message=last_message_text,
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
                user_prompt=user_input, message_history=message_history
            )

            messages = result.new_messages()

            tool_calls = {}

            for m in messages:
                for part in m.parts:
                    kind = part.part_kind

                    if kind == "text":
                        self.chat_interface.display_response(part.content)

                    if kind == "tool-call":
                        call_id = part.tool_call_id
                        tool_calls[call_id] = part

                    if kind == "tool-return":
                        call_id = part.tool_call_id
                        call = tool_calls[call_id]
                        result = part.content
                        self.chat_interface.display_function_call(
                            call.tool_name, json.dumps(call.args), result
                        )

            message_history.extend(messages)


class OpenAIChatCompletionsRunner(ChatRunner):
    """Runner for OpenAI chat completions API."""

    def __init__(
        self,
        tools: Tools,
        developer_prompt: str,
        chat_interface: ChatInterface,
        llm_client: LLMClient,
    ):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.llm_client = llm_client
        self.displaying_callback = DisplayingRunnerCallback(chat_interface)

    def convert_function_output_to_tool_message(self, data):
        return {
            "role": "tool",
            "tool_call_id": data["call_id"],
            "content": data["output"],
        }

    def loop(
        self,
        prompt: str,
        previous_messages: list = None,
        callback: RunnerCallback = None,
    ) -> LoopResult:
        chat_messages = []
        prev_messages_len = 0

        if previous_messages is None or len(previous_messages) == 0:
            chat_messages.append({"role": "system", "content": self.developer_prompt})
        else:
            chat_messages.extend(previous_messages)
            prev_messages_len = len(previous_messages)

        chat_messages.append({"role": "user", "content": prompt})

        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            reponse = self.llm_client.send_request(chat_messages, self.tools)

            if hasattr(reponse, "usage") and reponse.usage:
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
                function_call = dict(call.function.model_dump())
                function_call["call_id"] = call.id

                call_result = self.tools.function_call(function_call)
                call_result = self.convert_function_output_to_tool_message(call_result)

                chat_messages.append(call_result)

                if callback:
                    callback.on_function_call(function_call, call_result["content"])

        pricing_config = PricingConfig()
        cost_info = pricing_config.calculate_cost(
            self.llm_client.model, total_input_tokens, total_output_tokens
        )

        token_usage = TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )

        new_messages = chat_messages[prev_messages_len:]

        last_message_text = (message_response.content or "").strip()

        return LoopResult(
            new_messages=new_messages,
            all_messages=chat_messages,
            tokens=token_usage,
            cost=cost_info,
            last_message=last_message_text,
        )

    def run(self, stop_criteria: Callable = None) -> LoopResult:
        chat_messages = [
            {"role": "system", "content": self.developer_prompt},
        ]

        total_input_tokens = 0
        total_output_tokens = 0
        last_message_text = ""

        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended")
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

        pricing_config = PricingConfig()
        combined_cost = pricing_config.calculate_cost(
            self.llm_client.model, total_input_tokens, total_output_tokens
        )
        combined_tokens = TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )

        return LoopResult(
            new_messages=chat_messages,
            all_messages=chat_messages,
            tokens=combined_tokens,
            cost=combined_cost,
            last_message=last_message_text,
        )
