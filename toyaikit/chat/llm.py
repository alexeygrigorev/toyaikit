from openai import OpenAI
from toyaikit.tools import Tools
from typing import List

class LLMClient:
    def send_request(self, chat_messages: List, tools: Tools = None):
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", client: OpenAI = None):
        self.model = model

        if client is None:
            self.client = OpenAI()
        else:
            self.client = client

    def send_request(self, chat_messages: List, tools: Tools = None):
        tools_list = []
        if tools is not None:
            tools_list = tools.get_tools()

        return self.client.responses.create(
            model=self.model,
            input=chat_messages,
            tools=tools_list,
        )
