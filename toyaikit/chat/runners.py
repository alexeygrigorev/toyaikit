import uuid
import json
from typing import Callable
from abc import ABC, abstractmethod

from toyaikit.tools import Tools
from toyaikit.chat.ipython import ChatInterface
from toyaikit.llm import LLMClient


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


class ChatRunner(ABC):
    """Abstract base class for different chat runners."""

    def loop(
        self,
        prompt: str,
        previous_messages: list = None,
        callback: RunnerCallback = None,
    ) -> list:
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

    def loop(
        self,
        prompt: str,
        previous_messages: list[dict] = None,
        callback: RunnerCallback = None,
    ) -> list:
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

        while True:
            response = self.llm_client.send_request(
                chat_messages=chat_messages,
                tools=self.tools,
            )

            has_function_calls = False

            for entry in response.output:
                chat_messages.append(entry)

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

        return chat_messages[prev_messages_len:]

    def run(
        self,
        previous_messages: list = None,
        stop_criteria: Callable = None,
    ) -> list:
        if previous_messages is None or len(previous_messages) == 0:
            chat_messages = [
                {"role": "developer", "content": self.developer_prompt},
            ]
        else:
            chat_messages = []
            chat_messages.extend(previous_messages)

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            new_messages = self.loop(
                prompt=question,
                previous_messages=chat_messages,
                callback=self.displaying_callback,
            )

            chat_messages.extend(new_messages)
            if stop_criteria and stop_criteria(new_messages):
                break

        return chat_messages


class OpenAIAgentsSDKRunner(ChatRunner):
    """Runner for OpenAI Agents SDK."""

    def __init__(self, chat_interface: ChatInterface, agent):
        try:
            from agents import Runner, SQLiteSession
        except ImportError:
            raise ImportError(
                "Please run 'pip install openai-agents' to use this feature"
            )

        self.agent = agent
        self.runner = Runner()
        session_id = f"chat_session_{uuid.uuid4().hex[:8]}"
        self.session = SQLiteSession(session_id)
        self.chat_interface = chat_interface

    async def run(self) -> None:
        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            result = await self.runner.run(
                self.agent, input=user_input, session=self.session
            )

            func_calls = {}

            for ni in result.new_items:
                raw = ni.raw_item

                if ni.type == "tool_call_item":
                    func_calls[raw.call_id] = raw

                if ni.type == "tool_call_output_item":
                    func_call = func_calls[raw["call_id"]]
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


class D(dict):
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            return D(value)
        return value


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
    ) -> list:
        chat_messages = []
        prev_messages_len = 0

        if previous_messages is None or len(previous_messages) == 0:
            chat_messages.append({"role": "system", "content": self.developer_prompt})
        else:
            chat_messages.extend(previous_messages)
            prev_messages_len = len(previous_messages)

        chat_messages.append({"role": "user", "content": prompt})

        while True:
            reponse = self.llm_client.send_request(chat_messages, self.tools)

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
                function_call = D(call.function.model_dump())
                function_call["call_id"] = call.id

                call_result = self.tools.function_call(function_call)
                call_result = self.convert_function_output_to_tool_message(call_result)

                chat_messages.append(call_result)

                if callback:
                    callback.on_function_call(function_call, call_result["content"])

        return chat_messages[prev_messages_len:]

    def run(self) -> None:
        chat_messages = [
            {"role": "system", "content": self.developer_prompt},
        ]

        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended")
                break

            new_messages = self.loop(
                prompt=user_input,
                previous_messages=chat_messages,
                callback=self.displaying_callback,
            )

            chat_messages.append(new_messages)
