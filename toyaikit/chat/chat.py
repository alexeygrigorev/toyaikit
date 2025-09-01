from toyaikit.tools import Tools
from toyaikit.llm import LLMClient
from toyaikit.chat.interface import ChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner


class ChatAssistant:
    def __init__(
        self,
        tools: Tools,
        developer_prompt: str,
        chat_interface: ChatInterface,
        llm_client: LLMClient,
    ):
        self.runner = OpenAIResponsesRunner(
            tools=tools,
            developer_prompt=developer_prompt,
            chat_interface=chat_interface,
            llm_client=llm_client,
        )

    def run(self):
        self.runner.run()
