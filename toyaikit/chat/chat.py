from toyaikit.tools import Tools
from toyaikit.chat.ipython import ChatInterface, IPythonChatInterface
from toyaikit.chat.llm import LLMClient, OpenAIClient


class ChatAssistant:
    def __init__(
        self,
        tools: Tools,
        developer_prompt: str = "",
        chat_interface: ChatInterface = None,
        llm_client: LLMClient = None,
    ):
        self.tools = tools
        self.developer_prompt = developer_prompt

        if chat_interface is None:
            self.chat_interface = IPythonChatInterface()
        else:
            self.chat_interface = chat_interface

        if llm_client is None:
            self.llm_client = OpenAIClient()
        else:
            self.llm_client = llm_client

    def run(self):
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

                # Parse tool calls using the tools class
                tool_calls = self.tools.parse_response(response)
                
                # Add the assistant's response to chat history and display it
                assistant_message = self.tools.api_adapter.extract_assistant_message(response)
                chat_messages.append(assistant_message)
                
                if assistant_message.get("content"):
                    self.chat_interface.display_response(assistant_message["content"])

                # Handle tool calls if any
                for tool_call in tool_calls:
                    result = self.tools.function_call(tool_call)
                    chat_messages.append(result)
                    self.chat_interface.display_function_call(
                        tool_call.name, tool_call.arguments, result
                    )
                    has_function_calls = True

                if not has_function_calls:
                    break
