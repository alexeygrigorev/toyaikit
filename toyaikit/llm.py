from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.parsed_response import ParsedResponse

from anthropic.types import Message, RawMessageStopEvent

from toyaikit.tools import Tools


class LLMClient:
    def send_request(self, chat_messages: List, tools: Tools = None):
        raise NotImplementedError("Subclasses must implement this method")


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: OpenAI = None,
        extra_kwargs: dict = None,
    ):
        self.model = model

        if client is None:
            self.client = OpenAI()
        else:
            self.client = client

        self.extra_kwargs = extra_kwargs or {}

    def send_request(
        self,
        chat_messages: List,
        tools: Tools = None,
        output_format: BaseModel = None,
    ) -> Response | ParsedResponse:
        tools_list = []

        if tools is not None:
            tools_list = tools.get_tools()

        args = dict(
            model=self.model,
            input=chat_messages,
            tools=tools_list,
            **self.extra_kwargs,
        )

        if output_format is not None:
            return self.client.responses.parse(
                text_format=output_format,
                **args,
            )

        return self.client.responses.create(**args)


class OpenAIChatCompletionsClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: OpenAI = None,
        extra_kwargs: dict = None,
    ):
        self.model = model

        if client is None:
            self.client = OpenAI()
        else:
            self.client = client

        self.extra_kwargs = extra_kwargs or {}

    def convert_single_tool(self, tool, strict: bool = False):
        """
        Convert a single OpenAI tool/function API dict to Chat Completions function format.
        """
        fn = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        if strict:
            fn["function"]["strict"] = True
        return fn

    def convert_api_tools_to_chat_functions(self, api_tools, strict: bool = False):
        """
        Convert a list of OpenAI API tools to Chat Completions function format.
        """
        chat_functions = []

        for tool in api_tools:
            converted = self.convert_single_tool(tool, strict=strict)
            chat_functions.append(converted)

        return chat_functions

    def send_request(
        self,
        chat_messages: List,
        tools: Tools = None,
        output_format: BaseModel = None,
    ) -> ChatCompletion | ParsedChatCompletion:
        tools_list = []

        if tools is not None:
            tools_requests_format = tools.get_tools()

            strict = output_format is not None
            tools_list = self.convert_api_tools_to_chat_functions(
                tools_requests_format,
                strict=strict,
            )

        args = dict(
            model=self.model,
            messages=chat_messages,
            tools=tools_list,
            **self.extra_kwargs,
        )

        if output_format is not None:
            return self.client.chat.completions.parse(
                response_format=output_format,
                **args,
            )

        return self.client.chat.completions.create(**args)


class AnthropicClient(LLMClient):
    """Client for Anthropic's Messages API (Claude)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250514",
        api_key: str = None,
        base_url: str = None,
        extra_kwargs: dict = None,
    ):
        """
        Initialize Anthropic client.

        Args:
            model: Model name (e.g., "claude-sonnet-4-5-20250514")
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            base_url: Optional base URL for compatible APIs (e.g., z.ai)
            extra_kwargs: Additional kwargs to pass to messages.create
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Please run 'pip install anthropic' to use AnthropicClient"
            )

        self.model = model
        self.extra_kwargs = extra_kwargs or {}

        client_kwargs = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.client = Anthropic(**client_kwargs)

    def convert_openai_tool_to_anthropic(self, tool: dict) -> dict:
        """Convert OpenAI tool format to Anthropic tool format."""
        return {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }

    def send_request(
        self,
        chat_messages: List,
        tools: Tools = None,
        output_format: BaseModel = None,
    ) -> Message:
        """
        Send a request to Anthropic's Messages API.

        Args:
            chat_messages: List of message dictionaries with 'role' and 'content'
            tools: Optional Tools object with function definitions
            output_format: Optional Pydantic BaseModel for structured output

        Returns:
            Message response from Anthropic
        """
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None

        for msg in chat_messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                # Anthropic expects system message as a separate parameter
                if isinstance(content, str):
                    system_message = content
                elif isinstance(content, list):
                    # Handle list content (e.g., with cache_control)
                    system_message = content
            elif role in ("user", "assistant"):
                anthropic_messages.append({
                    "role": role,
                    "content": content if isinstance(content, (str, list)) else str(content)
                })
            elif role == "tool":
                # Convert tool response to user message format
                # Anthropic expects tool results inside a user message with tool_use content
                tool_id = msg.get("tool_call_id", "")
                tool_output = content if isinstance(content, str) else str(content)
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": tool_output
                        }
                    ]
                })

        # Prepare tools
        tools_list = None
        if tools is not None:
            openai_tools = tools.get_tools()
            tools_list = [
                self.convert_openai_tool_to_anthropic(tool)
                for tool in openai_tools
            ]

        # Build args - max_tokens is required by Anthropic API
        # Use extra_kwargs max_tokens if provided, otherwise default to 4096
        args = dict(
            model=self.model,
            messages=anthropic_messages,
            max_tokens=self.extra_kwargs.get("max_tokens", 4096),
        )

        # Add other extra_kwargs (excluding max_tokens which we already added)
        for key, value in self.extra_kwargs.items():
            if key != "max_tokens":
                args[key] = value

        if system_message is not None:
            args["system"] = system_message

        if tools_list is not None:
            args["tools"] = tools_list

        # Handle structured output
        if output_format is not None:
            # Use Anthropic's structured output feature
            # Note: This requires SDK version >= 0.40.0
            args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_format.__name__,
                    "strict": True,
                    "schema": output_format.model_json_schema(),
                },
            }

        return self.client.messages.create(**args)
