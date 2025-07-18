from openai import OpenAI

from toyaikit.tools import Tools
from toyaikit.chat.ipython import ChatInterface


class LLMClient:
    def send_request(self, chat_messages):
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIClient(LLMClient):
    def __init__(self, tools: Tools, model: str = "gpt-4o-mini", client: OpenAI = None):
        self.model = model
        self.tools = tools

        if client is None:
            self.client = OpenAI()
        else:
            self.client = client

    def send_request(self, chat_messages):
        return self.client.responses.create(
            model=self.model,
            input=chat_messages,
            tools=self.tools.get_tools(),
        )

class ChatAssistant:
    def __init__(self, tools: Tools, developer_prompt: str, chat_interface: ChatInterface, llm_client: LLMClient):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.llm_client = llm_client
    
    def run(self):
        chat_messages = [
            {"role": "developer", "content": self.developer_prompt},
        ]

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.lower() == 'stop':
                self.chat_interface.display("Chat ended.")
                break

            message = {"role": "user", "content": question}
            chat_messages.append(message)

            while True:  # inner request loop
                response = self.llm_client.send_request(chat_messages)

                has_messages = False

                for entry in response.output:
                    chat_messages.append(entry)

                    if entry.type == "function_call":
                        result = self.tools.function_call(entry)
                        chat_messages.append(result)
                        self.chat_interface.display_function_call(entry, result)

                    elif entry.type == "message":
                        self.chat_interface.display_response(entry)
                        has_messages = True

                if has_messages:
                    break 