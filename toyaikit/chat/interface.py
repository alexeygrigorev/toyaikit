import mistune
from IPython.display import HTML
from IPython.display import display as ip_display


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class ChatInterface:
    def input(self) -> str:
        """
        Get input from the user.
        Returns:
            str: The user's input.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def display(self, message: str) -> None:
        """
        Display a message.
        Args:
            message: The message to display.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def display_function_call(
        self, function_name: str, arguments: str, result: str
    ) -> None:
        """
        Display a function call.
        Args:
            function_name: The name of the function to call.
            arguments: The arguments to pass to the function.
            result: The result of the function call.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def display_response(self, markdown_text: str) -> None:
        """
        Display a response.
        Args:
            markdown_text: The markdown text to display.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def display_reasoning(self, markdown_text: str) -> None:
        """
        Display a reasoning.
        Args:
            markdown_text: The markdown text to display.
        """
        raise NotImplementedError("Subclasses must implement this method")


class IPythonChatInterface:
    def input(self) -> str:
        question = input("You:")
        return question.strip()

    def display(self, message: str) -> None:
        print(message)

    def display_function_call(
        self, function_name: str, arguments: str, result: str
    ) -> None:
        excaped_result = result.replace("<", "&lt;").replace(">", "&gt;")
        call_html = f"""
            <details>
            <summary>Function call: <tt>{function_name}({shorten(arguments)})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{arguments}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{excaped_result}</pre>
            </div>

            </details>
        """
        ip_display(HTML(call_html))

    def display_reasoning(self, markdown_text: str) -> None:
        reasoning_html = mistune.html(markdown_text)
        html = f"""
            <details>
                <summary>Reasoning</summary>
                <div>{reasoning_html}</div>
            </details>
        """
        ip_display(HTML(html))

    def display_response(self, markdown_text: str) -> None:
        response_html = mistune.html(markdown_text)
        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{response_html}</div>
            </div>
        """
        ip_display(HTML(html))


class StdOutputInterface:
    def input(self) -> str:
        """
        Get input from the user.
        Returns:
            str: The user's input.
        """
        question = input("You: ")
        return question.strip()

    def display(self, message: str) -> None:
        """
        Display a message.
        Args:
            message: The message to display.
        """
        print(message)

    def display_function_call(
        self, function_name: str, arguments: str, result: str
    ) -> None:
        """
        Display a function call.
        Args:
            function_name: The name of the function to call.
            arguments: The arguments to pass to the function.
            result: The result of the function call.
        """
        print()
        print("--- Function Call ---")
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")
        print(f"Result: {result}")
        print("-------------------")
        print()

    def display_response(self, markdown_text: str) -> None:
        """
        Display a response.
        Args:
            markdown_text: The markdown text to display.
        """
        print()
        print(f"Assistant: {markdown_text}\n")

    def display_reasoning(self, markdown_text: str) -> None:
        """
        Display a reasoning.
        Args:
            markdown_text: The markdown text to display.
        """
        print()
        print("--- Reasoning ---")
        print(markdown_text)
        print("---------------")
        print()
