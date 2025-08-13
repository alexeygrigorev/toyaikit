import json
import uuid

from toyaikit.tools import Tools
from toyaikit.tools import generate_function_schema, generate_schemas_from_instance
from toyaikit.tools import python_type_to_json_type

class ToolCallResponse:
    def __init__(self, name, arguments, type="function_call", status="completed"):
        self.name = name
        self.arguments = arguments
        self.call_id = uuid.uuid4().hex
        self.type = type
        self.id = uuid.uuid4().hex
        self.status = status

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

def test_add_tools_and_generate_schemas_from_instance():
    class Dummy:
        def foo(self, x: int) -> int:
            """Return x+1"""
            return x + 1
        def bar(self, y: str) -> str:
            return y.upper()

    dummy = Dummy()
    tools = Tools()
    tools.add_tools(dummy)

    tool_names = {tool["name"] for tool in tools.get_tools()}
    assert tool_names == {"foo", "bar"}

    # Check schema docstring for foo
    foo_schema = [t for t in tools.get_tools() if t["name"] == "foo"][0]
    assert "Return x+1" in foo_schema["description"]


    resp = ToolCallResponse("foo", json.dumps({"x": 41}))
    result = tools.function_call(resp)
    assert json.loads(result["output"]) == 42


def test_generate_function_schema_various_cases():
    # With docstring and type hints
    def f1(a: int, b: str) -> str:
        """Docstring here"""
        return b * a

    schema1 = generate_function_schema(f1)
    assert schema1["description"].startswith("Docstring")
    assert schema1["parameters"]["properties"]["a"]["type"] == "number"
    assert schema1["parameters"]["properties"]["b"]["type"] == "string"

    # No docstring, no type hints
    def f2(x):
        return x

    schema2 = generate_function_schema(f2)
    assert schema2["description"] == "No description provided."

    # Default value not required
    def f3(a: int = 5):
        return a

    schema3 = generate_function_schema(f3)
    assert "a" in schema3["parameters"]["properties"]
    assert "a" not in schema3["parameters"]["required"]


def test_python_type_to_json_type():
    assert python_type_to_json_type(str) == "string"
    assert python_type_to_json_type(int) == "number"
    assert python_type_to_json_type(float) == "number"
    assert python_type_to_json_type(bool) == "boolean"
    assert python_type_to_json_type(list) == "array"
    assert python_type_to_json_type(dict) == "object"
    class Custom: pass
    assert python_type_to_json_type(Custom) == "string"


def test_generate_schemas_from_instance_skips_private():
    class Dummy:
        def _private(self): pass
        def public(self): return 1

    dummy = Dummy()

    schemas = generate_schemas_from_instance(dummy)
    names = [f.__name__ for f, _ in schemas]

    assert "public" in names
    assert "_private" not in names


def test_function_call_errors():

    def foo(a: int):
        return a

    tools = Tools()
    tools.add_tool(foo)
    
    # Unknown tool
    bad_resp = ToolCallResponse("not_a_tool", json.dumps({}))
    result = tools.function_call(bad_resp)
    assert result["type"] == "function_call_output"
    assert result["call_id"] == bad_resp.call_id
    error_data = json.loads(result["output"])
    assert "error" in error_data
    assert "KeyError" in error_data["error"]
    
    # Bad arguments
    bad_args = ToolCallResponse("foo", json.dumps({"b": 1}))
    result = tools.function_call(bad_args)
    assert result["type"] == "function_call_output"
    assert result["call_id"] == bad_args.call_id
    error_data = json.loads(result["output"])
    assert "error" in error_data
    assert "TypeError" in error_data["error"]


def test_add_tool_auto_schema():
    def bar(x: int) -> int:
        """Returns x squared"""
        return x * x

    tools = Tools()
    tools.add_tool(bar)  # No schema provided

    print(tools.get_tools())
    
    tool = [t for t in tools.get_tools() if t["name"] == "bar"][0]
    assert tool["description"].startswith("Returns x squared")

    # Test function_call
    resp = ToolCallResponse("bar", json.dumps({"x": 4}))
    result = tools.function_call(resp)
    assert json.loads(result["output"]) == 16


def test_auto_schema_get_tools_output():
    def bar(x: int) -> int:
        """Returns x squared"""
        return x * x

    tools = Tools()
    tools.add_tool(bar)  # No schema provided

    expected = [{
        'type': 'function',
        'name': 'bar',
        'description': 'Returns x squared',
        'parameters': {
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'description': 'x parameter'}
            },
            'required': ['x'],
            'additionalProperties': False
        }
    }]

    assert tools.get_tools() == expected


def test_auto_schema_get_tools_output_class_instance_multiple_methods():
    class MyClass:
        def bar(self, x: int) -> int:
            """Returns x squared"""
            return x * x
        def foo(self, y: float) -> float:
            """Returns y plus 1"""
            return y + 1

    obj = MyClass()
    tools = Tools()
    tools.add_tools(obj)

    tools_list = tools.get_tools()
    tool_names = {t['name'] for t in tools_list}
    assert 'bar' in tool_names
    assert 'foo' in tool_names

    bar_schema = next(t for t in tools_list if t['name'] == 'bar')
    foo_schema = next(t for t in tools_list if t['name'] == 'foo')

    expected_bar = {
        'type': 'function',
        'name': 'bar',
        'description': 'Returns x squared',
        'parameters': {
            'type': 'object',
            'properties': {
                'x': {'type': 'number', 'description': 'x parameter'}
            },
            'required': ['x'],
            'additionalProperties': False
        }
    }
    expected_foo = {
        'type': 'function',
        'name': 'foo',
        'description': 'Returns y plus 1',
        'parameters': {
            'type': 'object',
            'properties': {
                'y': {'type': 'number', 'description': 'y parameter'}
            },
            'required': ['y'],
            'additionalProperties': False
        }
    }
    assert bar_schema == expected_bar
    assert foo_schema == expected_foo