from openai import OpenAI
from toyaikit.tools import Tools
from typing import List
import requests
import json

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

class RequestsChatCompletionsClient(LLMClient):
    def __init__(self, url: str, api_key: str, model: str):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def send_request(self, chat_messages: List, tools: Tools = None):
        request_data = {
            "model": self.model,
            "messages": chat_messages,
        }
        
        if tools is not None:
            tools_list = tools.get_tools()
            request_data["tools"] = tools_list

        response = requests.post(self.url, headers=self.headers, json=request_data)
        response.raise_for_status()
        
        return response.json()
