import json
import uuid
from toyaikit.tools import Tools

# Define simple tool functions and their descriptions for testing

def add(a: float, b: float) -> float:
    return a + b

add_tool_desc = {
    "type": "function",
    "name": "add",
    "description": "Add two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number."},
            "b": {"type": "number", "description": "Second number."}
        },
        "required": ["a", "b"],
        "additionalProperties": False
    }
}

def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool_desc = {
    "type": "function",
    "name": "multiply",
    "description": "Multiply two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number."},
            "b": {"type": "number", "description": "Second number."}
        },
        "required": ["a", "b"],
        "additionalProperties": False
    }
}

def echo(text: str) -> str:
    return text

echo_tool_desc = {
    "type": "function",
    "name": "echo",
    "description": "Echo the input text.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo back."}
        },
        "required": ["text"],
        "additionalProperties": False
    }
}

def test_tools_registration_and_call():
    tools = Tools()
    tools.add_tool(add, add_tool_desc)
    tools.add_tool(multiply, multiply_tool_desc)
    tools.add_tool(echo, echo_tool_desc)

    # Check get_tools returns the correct descriptions
    tool_names = {tool["name"] for tool in tools.get_tools()}
    assert tool_names == {"add", "multiply", "echo"}

    # Simulate a tool_call_response object with all real fields
    class ToolCallResponse:
        def __init__(self, name, arguments, type="function_call", status="completed"):
            self.name = name
            self.arguments = arguments
            self.call_id = uuid.uuid4().hex
            self.type = type
            self.id = uuid.uuid4().hex
            self.status = status

    # Test add
    add_args = json.dumps({"a": 2, "b": 3})
    add_resp = ToolCallResponse("add", add_args)
    result = tools.function_call(add_resp)
    assert json.loads(result["output"]) == 5

    # Test multiply
    mul_args = json.dumps({"a": 4, "b": 5})
    mul_resp = ToolCallResponse("multiply", mul_args)
    result = tools.function_call(mul_resp)
    assert json.loads(result["output"]) == 20

    # Test echo
    echo_args = json.dumps({"text": "hello"})
    echo_resp = ToolCallResponse("echo", echo_args)
    result = tools.function_call(echo_resp)
    assert json.loads(result["output"]) == "hello" 