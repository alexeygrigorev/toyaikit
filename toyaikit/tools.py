import json
import inspect

from typing import get_type_hints


class Tools:
    def __init__(self):
        self.tools = {}
        self.functions = {}

    def add_tool(self, function, schema=None):
        """
        Add a tool to the Tools object.

        For the schema, this is the format:

        {
            "type": "function",
            "name": "<function_name>",
            "description": "<Brief explanation of what the function does>",
            "parameters": {
                "type": "object",
                "properties": {
                "<parameter_name>": {
                    "type": "<data_type>",  // e.g., "string", "number", "boolean", "array", "object"
                    "description": "<Description of the parameter's purpose>"
                }
                // Additional parameters can be added here
                },
                "required": ["<param1>", "<param2>"],  // List all required parameters here
                "additionalProperties": false
            }
        }


        Args:
            function: The function to add as a tool.
            schema: The schema of the function. If not provided, it will be generated automatically.

        """
        if schema is None:
            schema = generate_function_schema(function)
        self.tools[function.__name__] = schema
        self.functions[function.__name__] = function

    def add_tools(self, instance):
        """
        Add all tools from an instance.
        """
        instance_tools = generate_schemas_from_instance(instance)
        for function, schema in instance_tools:
            self.add_tool(function, schema)

    def get_tools(self):
        """
        Get the tools in the Tools object.

        Returns:
            list: A list of tools in the Tools object.
        """
        return list(self.tools.values())

    def function_call(self, tool_call_response):
        """
        Handle a function call from the LLM.

        Args:
            tool_call_response: The tool call response from the LLM.

        Returns:
            dict: The result of the function call.
        """

        function_name = tool_call_response.name
        arguments = json.loads(tool_call_response.arguments)
        f = self.functions[function_name]
        result = f(**arguments)
        return {
            "type": "function_call_output",
            "call_id": tool_call_response.call_id,
            "output": json.dumps(result, indent=2),
        } 


def generate_function_schema(func, description=None):
    """
    Generate a schema for a function.

    Args:
        func: The function to generate a schema for.
    """

    sig = inspect.signature(func)
    hints = get_type_hints(func)

    if description is None:
        doc = inspect.getdoc(func)
        if doc is None:
            description = "No description provided."
        else:
            description = doc.strip()

    schema = {
        "type": "function",
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    }
    
    for name, param in sig.parameters.items():
        param_type = hints.get(name, str)
        json_type = python_type_to_json_type(param_type)
        schema["parameters"]["properties"][name] = {
            "type": json_type,
            "description": f"{name} parameter"  # You can enhance this with more info
        }
        if param.default is inspect.Parameter.empty:
            schema["parameters"]["required"].append(name)
    
    return schema

def python_type_to_json_type(py_type):
    """
    Convert a Python type to a JSON type.

    Args:
        py_type: The Python type to convert.
    """

    if py_type in [str]:
        return "string"
    elif py_type in [int, float]:
        return "number"
    elif py_type == bool:
        return "boolean"
    elif py_type == list:
        return "array"
    elif py_type == dict:
        return "object"
    else:
        return "string"  # fallback for unknown types


def generate_schemas_from_instance(instance):
    """
    Generate schemas for all methods in an instance.

    Args:
        instance: The instance to generate schemas for.

    Returns:
        list: A list of tuples, each containing a function and its schema.
    """

    instance_tools = []
    for name, member in inspect.getmembers(instance, predicate=inspect.ismethod):
        if name.startswith("_"):
            continue
        schema = generate_function_schema(member)
        instance_tools.append((member, schema))
    return instance_tools