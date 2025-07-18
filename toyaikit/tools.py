import json


class Tools:
    def __init__(self):
        self.tools = {}
        self.functions = {}

    def add_tool(self, function, description):
        self.tools[function.__name__] = description
        self.functions[function.__name__] = function
    
    def get_tools(self):
        return list(self.tools.values())

    def function_call(self, tool_call_response):
        function_name = tool_call_response.name
        arguments = json.loads(tool_call_response.arguments)
        f = self.functions[function_name]
        result = f(**arguments)
        return {
            "type": "function_call_output",
            "call_id": tool_call_response.call_id,
            "output": json.dumps(result, indent=2),
        } 