from typing import List

from openai import OpenAI
from pydantic import BaseModel

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.parsed_response import ParsedResponse



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

    def convert_single_tool(self, tool):
        """
        Convert a single OpenAI tool/function API dict to Chat Completions function format.
        """
        if tool["type"] != "function":
            raise "it's not a function"

        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }

    def convert_api_tools_to_chat_functions(self, api_tools):
        """
        Convert a list of OpenAI API tools to Chat Completions function format.
        """
        chat_functions = []

        for tool in api_tools:
            converted = self.convert_single_tool(tool)
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
            tools_list = self.convert_api_tools_to_chat_functions(tools_requests_format)

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
