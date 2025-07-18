
class ChatAssistant:
    def __init__(self, tools, developer_prompt, chat_interface, client):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
        self.client = client
    
    def gpt(self, chat_messages):
        return self.client.responses.create(
            model='gpt-4o-mini',
            input=chat_messages,
            tools=self.tools.get_tools(),
        )

    def run(self):
        chat_messages = [
            {"role": "developer", "content": self.developer_prompt},
        ]

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.strip().lower() == 'stop':  
                self.chat_interface.display("Chat ended.")
                break

            message = {"role": "user", "content": question}
            chat_messages.append(message)

            while True:  # inner request loop
                response = self.gpt(chat_messages)

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