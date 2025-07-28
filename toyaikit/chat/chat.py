from toyaikit.tools import Tools
from toyaikit.chat.ipython import ChatInterface
from toyaikit.chat.llm import LLMClient

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
                response = self.llm_client.send_request(chat_messages, self.tools)

                has_function_calls = False

                for entry in response.output:
                    chat_messages.append(entry)

                    if entry.type == "function_call":
                        result = self.tools.function_call(entry)
                        chat_messages.append(result)
                        self.chat_interface.display_function_call(entry.name, entry.arguments, result)
                        has_function_calls = True

                    elif entry.type == "message":
                        markdown_text = entry.content[0].text
                        self.chat_interface.display_response(markdown_text)

                if not has_function_calls:
                    break 