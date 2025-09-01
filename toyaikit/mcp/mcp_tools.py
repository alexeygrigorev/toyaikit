import json


def convert_mcp_tool_to_function_format(mcp_tool):
    """
    Convert MCP tool format to function format.
    
    Args:
        mcp_tool: Tool object or dict with MCP format
    
    Returns:
        dict: Tool in function format
    """
    # Handle both Tool objects and dictionaries
    if hasattr(mcp_tool, 'name'):
        # It's a Tool object
        name = mcp_tool.name
        description = mcp_tool.description
        input_schema = mcp_tool.inputSchema
    else:
        # It's a dictionary
        name = mcp_tool['name']
        description = mcp_tool['description']
        input_schema = mcp_tool['inputSchema']
    
    # Clean up description - remove docstring formatting
    clean_description = description.split('\n\n')[0] if '\n\n' in description else description
    clean_description = clean_description.strip()
    
    # Convert the tool format
    function_tool = {
        "type": "function",
        "name": name,
        "description": clean_description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": input_schema.get('required', []),
            "additionalProperties": False
        }
    }
    
    # Convert properties
    if 'properties' in input_schema:
        for prop_name, prop_info in input_schema['properties'].items():
            function_tool["parameters"]["properties"][prop_name] = {
                "type": prop_info.get('type', 'string'),
                "description": prop_info.get('description', f"{prop_name.replace('_', ' ').title()}")
            }
            
            # Add title as description if no description exists
            if 'title' in prop_info and 'description' not in prop_info:
                function_tool["parameters"]["properties"][prop_name]["description"] = prop_info['title']
    
    return function_tool


def convert_tools_list(mcp_tools):
    """
    Convert a list of MCP tools to function format.
    
    Args:
        mcp_tools: List of MCP tools
    
    Returns:
        list: List of tools in function format
    """
    return [convert_mcp_tool_to_function_format(tool) for tool in mcp_tools]



class MCPTools:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.tools = None
    
    def get_tools(self):
        if self.tools is None:
            mcp_tools = self.mcp_client.get_tools()
            self.tools = convert_tools_list(mcp_tools)
        return self.tools

    def function_call(self, tool_call_response):
        function_name = tool_call_response.name
        arguments = json.loads(tool_call_response.arguments)

        result = self.mcp_client.call_tool(function_name, arguments)
        output = result['content'][0]['text']

        return {
            "type": "function_call_output",
            "call_id": tool_call_response.call_id,
            "output": output,
        }