from IPython.display import display, HTML
import markdown


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


class ChatInterface:
    def input(self):
        raise NotImplementedError("Subclasses must implement this method")

    def display(self, message):
        raise NotImplementedError("Subclasses must implement this method")
    
    def display_function_call(self, entry, result):
        raise NotImplementedError("Subclasses must implement this method")
    
    def display_response(self, entry):
        raise NotImplementedError("Subclasses must implement this method")


class IPythonChatInterface:
    def input(self):
        question = input("You:")
        return question.strip()
    
    def display(self, message):
        print(message)

    def display_function_call(self, entry, result):
        call_html = f"""
            <details>
            <summary>Function call: <tt>{entry.name}({shorten(entry.arguments)})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{entry}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result['output']}</pre>
            </div>
            
            </details>
        """
        display(HTML(call_html))

    def display_response(self, entry):
        response_html = markdown.markdown(entry.content[0].text)
        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{response_html}</div>
            </div>
        """
        display(HTML(html)) 