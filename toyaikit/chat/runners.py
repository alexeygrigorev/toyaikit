import uuid
import json
from abc import ABC, abstractmethod

from toyaikit.tools import Tools
from toyaikit.chat.ipython import ChatInterface
from toyaikit.llm import LLMClient


class ChatRunner(ABC):
    """Abstract base class for different chat runners."""

    @abstractmethod
    def run(self) -> None:
        """
        Run the chat.
        """
        pass


class OpenAIResponsesRunner(ChatRunner):
    """Runner for OpenAI responses API."""

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

    def run(self) -> None:
        chat_messages = [
            {"role": "developer", "content": self.developer_prompt},
        ]

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.lower() == "stop":
                self.chat_interface.display("Chat ended.")
                break

            message = {"role": "user", "content": question}
            chat_messages.append(message)

            while True:  # inner request loop
                response = self.llm_client.send_request(chat_messages, self.tools)

                has_function_calls = False

                for entry in response.output:
                    chat_messages.append(entry)

                    if entry.type == "function_call":
                        result = self.tools.function_call(entry)
                        chat_messages.append(result)
                        self.chat_interface.display_function_call(
                            entry.name, entry.arguments, result
                        )
                        has_function_calls = True

                    elif entry.type == "message":
                        markdown_text = entry.content[0].text
                        self.chat_interface.display_response(markdown_text)

                if not has_function_calls:
                    break


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

    def convert_function_output_to_tool_message(self, data):
        return {
            "role": "tool",
            "tool_call_id": data["call_id"],
            "content": data["output"],
        }

    def run(self) -> None:
        chat_messages = [
            {"role": "system", "content": self.developer_prompt},
        ]

        while True:
            user_input = self.chat_interface.input()
            if user_input.lower() == "stop":
                self.chat_interface.display("Chat ended")
                break

            chat_messages.append({"role": "user", "content": user_input})

            while True:
                reponse = self.llm_client.send_request(chat_messages, self.tools)

                first_choice = reponse.choices[0]
                message_response = first_choice.message
                chat_messages.append(message_response)

                if hasattr(message_response, "reasoning_content"):
                    reasoning = message_response.reasoning_content.strip()
                    if reasoning != "":
                        self.chat_interface.display_reasoning(reasoning)

                content = message_response.content.strip()
                if content != "":
                    self.chat_interface.display_response(content)

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
                    call_result = self.convert_function_output_to_tool_message(
                        call_result
                    )

                    chat_messages.append(call_result)

                    self.chat_interface.display_function_call(
                        function_name=function_call.name,
                        arguments=function_call["arguments"],
                        result=call_result["content"],
                    )
