import json
import inspect
from abc import ABC, abstractmethod
from typing import get_type_hints


class ToolInterface(ABC):
    """Abstract interface for tool API adapters."""
    
    @abstractmethod
    def format_tools_for_api(self, tools_list):
        """Format tools list for the specific API."""
        pass
    
    @abstractmethod
    def parse_tool_calls(self, response):
        """Parse tool calls from API response."""
        pass
    
    @abstractmethod
    def format_tool_response(self, call_id, output):
        """Format tool response for the API."""
        pass
    
    @abstractmethod
    def extract_assistant_message(self, response):
        """Extract assistant message from API response."""
        pass


class ToolsResponseAPI(ToolInterface):
    """Adapter for the responses API."""
    
    def format_tools_for_api(self, tools_list):
        """Format tools for responses API - return as is."""
        return tools_list
    
    def parse_tool_calls(self, response):
        """Parse tool calls from responses API response."""
        # For responses API, tool calls are in response.tool_calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return response.tool_calls
        
        # Handle test format where tool calls are in response.output
        if hasattr(response, 'output'):
            tool_calls = []
            for entry in response.output:
                if hasattr(entry, 'type') and entry.type == 'function_call':
                    tool_calls.append(entry)
            return tool_calls
        
        return []
    
    def format_tool_response(self, call_id, output):
        """Format tool response for responses API."""
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(output, indent=2),
        }
    
    def extract_assistant_message(self, response):
        """Extract assistant message from responses API response."""
        # Handle test format where messages are in response.output
        if hasattr(response, 'output'):
            for entry in response.output:
                if hasattr(entry, 'type') and entry.type == "message":
                    if hasattr(entry, 'content') and entry.content:
                        return {
                            "role": "assistant",
                            "content": entry.content[0].text
                        }
            return {"role": "assistant", "content": None}
        
        # Handle real responses API format
        for entry in response.output:
            if entry.type == "message":
                return {
                    "role": "assistant",
                    "content": entry.content[0].text
                }
        return {"role": "assistant", "content": None}


class ToolsChatCompletionAPI(ToolInterface):
    """Adapter for the chat completions API."""

    def format_tools_for_api(self, tools_list):
        """Format tools for chat completions API - wrap each tool in function format."""
        formatted_tools = []
        for tool in tools_list:
            formatted_tool = {
                "type": "function",
                "function": tool
            }
            formatted_tools.append(formatted_tool)
        return formatted_tools
    
    def parse_tool_calls(self, response):
        """Parse tool calls from chat completions API response."""
        # For chat completions API, tool calls are in choices[0].message.tool_calls
        if (hasattr(response, 'choices') and 
            response.choices and 
            hasattr(response.choices[0], 'message') and
            hasattr(response.choices[0].message, 'tool_calls')):
            return response.choices[0].message.tool_calls
        return []
    
    def format_tool_response(self, call_id, output):
        """Format tool response for chat completions API."""
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(output, indent=2),
        }
    
    def extract_assistant_message(self, response):
        """Extract assistant message from chat completions API response."""
        return {
            "role": "assistant",
            "content": response["choices"][0]["message"]["content"]
        }


class Tools:
    def __init__(self, api='responses'):
        """
        Initialize Tools with specified API interface.
        
        Args:
            api: API type - 'responses' or 'chat.completions'
        """
        self.tools = {}
        self.functions = {}
        
        if api == 'responses':
            self.api_adapter = ToolsResponseAPI()
        elif api == 'chat.completions':
            self.api_adapter = ToolsChatCompletionAPI()
        else:
            raise ValueError(f"Unsupported API type: {api}. Use 'responses' or 'chat.completions'")

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
        Get tools formatted for the specific API.
        
        Returns:
            The tools formatted according to the API adapter.
        """
        tools_list = list(self.tools.values())
        return self.api_adapter.format_tools_for_api(tools_list)

    def function_call(self, tool_call_response):
        """
        Handle a function call from the LLM.

        Args:
            tool_call_response: The tool call response from the LLM.

        Returns:
            dict: The result of the function call formatted for the API.
        """
        function_name = tool_call_response.name
        arguments = json.loads(tool_call_response.arguments)
        f = self.functions[function_name]
        result = f(**arguments)
        return self.api_adapter.format_tool_response(tool_call_response.call_id, result)

    def parse_response(self, response):
        """
        Parse tool calls from an API response.
        
        Args:
            response: The API response object.
            
        Returns:
            list: List of tool calls to process.
        """
        return self.api_adapter.parse_tool_calls(response)


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