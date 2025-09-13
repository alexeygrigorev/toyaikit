from openai import OpenAI

from toyaikit.chat.chat import ChatAssistant
from toyaikit.chat.interface import IPythonChatInterface
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools


def init(
    developer_prompt: str, model: str = "gpt-4o-mini", client: OpenAI = None
) -> ChatAssistant:
    tools = Tools()

    if client is None:
        client = OpenAI()
    llm_client = OpenAIClient(model, client)

    chat_interface = IPythonChatInterface()

    chat_assistant = ChatAssistant(tools, developer_prompt, chat_interface, llm_client)

    return chat_assistant
