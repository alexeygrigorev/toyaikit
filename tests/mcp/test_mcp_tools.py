from unittest.mock import Mock

from toyaikit.mcp.mcp_tools import (
    MCPTools,
    convert_mcp_tool_to_function_format,
    convert_tools_list,
)


class TestConvertMcpToolToFunctionFormat:
    def test_convert_dictionary_tool_basic(self):
        """Test converting basic MCP tool dictionary to function format."""
        mcp_tool = {
            "name": "search",
            "description": "Search the FAQ database for entries matching the given query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text to look up in the course FAQ.",
                    }
                },
                "required": ["query"],
            },
        }

        result = convert_mcp_tool_to_function_format(mcp_tool)

        expected = {
            "type": "function",
            "name": "search",
            "description": "Search the FAQ database for entries matching the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text to look up in the course FAQ.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }

        assert result == expected

    def test_convert_object_tool_basic(self):
        """Test converting MCP tool object to function format."""

        # Create a simple object with the required attributes (like FastMCP tools)
        class MCPTool:
            def __init__(self):
                self.name = "add_entry"
                self.description = "Add a new entry to the FAQ database.\n\nArgs:\n    question (str): The question to be added to the FAQ database.\n    answer (str): The corresponding answer to the question."
                self.inputSchema = {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to be added to the FAQ database.",
                        },
                        "answer": {
                            "type": "string",
                            "description": "The corresponding answer to the question.",
                        },
                    },
                    "required": ["question", "answer"],
                }

        mcp_tool = MCPTool()

        result = convert_mcp_tool_to_function_format(mcp_tool)

        expected = {
            "type": "function",
            "name": "add_entry",
            "description": "Add a new entry to the FAQ database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to be added to the FAQ database.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The corresponding answer to the question.",
                    },
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        }

        assert result == expected

    def test_convert_tool_with_multiline_description(self):
        """Test converting tool with multiline description (should take first part)."""
        mcp_tool = {
            "name": "complex_function",
            "description": "Add a new entry to the FAQ database.\n\nArgs:\n    question (str): The question to be added\n    answer (str): The answer",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

        result = convert_mcp_tool_to_function_format(mcp_tool)

        assert result["description"] == "Add a new entry to the FAQ database."

    def test_convert_tool_without_required_fields(self):
        """Test converting tool without required fields in schema."""
        mcp_tool = {
            "name": "optional_function",
            "description": "Function with optional params",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "optional_param": {
                        "type": "string",
                        "description": "Optional parameter",
                    }
                },
            },
        }

        result = convert_mcp_tool_to_function_format(mcp_tool)

        assert result["parameters"]["required"] == []

    def test_convert_tool_with_title_fallback(self):
        """Test converting tool that uses title as description fallback."""
        mcp_tool = {
            "name": "titled_function",
            "description": "Function with titled params",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param_with_title": {
                        "type": "string",
                        "title": "Parameter With Title",
                    },
                    "param_without_desc": {"type": "integer"},
                },
                "required": [],
            },
        }

        result = convert_mcp_tool_to_function_format(mcp_tool)

        # Should use title as description
        assert (
            result["parameters"]["properties"]["param_with_title"]["description"]
            == "Parameter With Title"
        )
        # Should generate description from name
        assert (
            result["parameters"]["properties"]["param_without_desc"]["description"]
            == "Param Without Desc"
        )

    def test_convert_tool_with_no_properties(self):
        """Test converting tool with no properties in schema."""
        mcp_tool = {
            "name": "no_params_function",
            "description": "Function with no parameters",
            "inputSchema": {"type": "object", "required": []},
        }

        result = convert_mcp_tool_to_function_format(mcp_tool)

        assert result["parameters"]["properties"] == {}
        assert result["parameters"]["required"] == []

    def test_convert_tool_with_various_types(self):
        """Test converting tool with various parameter types."""
        mcp_tool = {
            "name": "multi_type_function",
            "description": "Function with multiple parameter types",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "string_param": {
                        "type": "string",
                        "description": "String parameter",
                    },
                    "number_param": {
                        "type": "number",
                        "description": "Number parameter",
                    },
                    "boolean_param": {
                        "type": "boolean",
                        "description": "Boolean parameter",
                    },
                    "array_param": {
                        "type": "array",
                        "description": "Array parameter",
                    },
                    "object_param": {
                        "type": "object",
                        "description": "Object parameter",
                    },
                    "no_type_param": {"description": "Parameter without type"},
                },
                "required": ["string_param", "number_param"],
            },
        }

        result = convert_mcp_tool_to_function_format(mcp_tool)

        # Check all types are preserved
        props = result["parameters"]["properties"]
        assert props["string_param"]["type"] == "string"
        assert props["number_param"]["type"] == "number"
        assert props["boolean_param"]["type"] == "boolean"
        assert props["array_param"]["type"] == "array"
        assert props["object_param"]["type"] == "object"
        # Should default to string when no type specified
        assert props["no_type_param"]["type"] == "string"

    def test_convert_tool_strips_whitespace_in_description(self):
        """Test that description whitespace is properly stripped."""
        mcp_tool = {
            "name": "whitespace_function",
            "description": "  \n  Function with whitespace  \n  ",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

        result = convert_mcp_tool_to_function_format(mcp_tool)

        assert result["description"] == "Function with whitespace"


class TestConvertToolsList:
    def test_convert_empty_list(self):
        """Test converting empty tools list."""
        result = convert_tools_list([])
        assert result == []

    def test_convert_search_and_add_entry_tools(self):
        """Test converting typical FAQ tools (search and add_entry)."""
        mcp_tools = [
            {
                "name": "search",
                "description": "Search the FAQ database for entries matching the given query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "add_entry",
                "description": "Add a new entry to the FAQ database.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question",
                        },
                        "answer": {
                            "type": "string",
                            "description": "The answer",
                        },
                    },
                    "required": ["question", "answer"],
                },
            },
        ]

        result = convert_tools_list(mcp_tools)

        assert len(result) == 2
        assert result[0]["name"] == "search"
        assert result[1]["name"] == "add_entry"
        assert all(tool["type"] == "function" for tool in result)

    def test_convert_multiple_tools_list(self):
        """Test converting list with multiple tools."""
        mcp_tools = [
            {
                "name": "function_one",
                "description": "First function",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "function_two",
                "description": "Second function",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

        result = convert_tools_list(mcp_tools)

        assert len(result) == 2
        assert result[0]["name"] == "function_one"
        assert result[1]["name"] == "function_two"
        assert all(tool["type"] == "function" for tool in result)


class TestMCPTools:
    def test_initialization(self):
        """Test MCPTools initialization."""
        mock_client = Mock()
        mcp_tools = MCPTools(mock_client)

        assert mcp_tools.mcp_client == mock_client
        assert mcp_tools.tools is None

    def test_get_tools_first_call(self):
        """Test get_tools() first call fetches from client."""
        mock_client = Mock()
        mock_client.get_tools.return_value = [
            {
                "name": "search",
                "description": "Search the FAQ database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]

        mcp_tools = MCPTools(mock_client)
        result = mcp_tools.get_tools()

        # Should call client once
        mock_client.get_tools.assert_called_once()

        # Should return converted tools
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["type"] == "function"

    def test_get_tools_cached(self):
        """Test get_tools() subsequent calls use cache."""
        mock_client = Mock()
        mock_client.get_tools.return_value = []

        mcp_tools = MCPTools(mock_client)

        # Call twice
        result1 = mcp_tools.get_tools()
        result2 = mcp_tools.get_tools()

        # Should only call client once (cached)
        mock_client.get_tools.assert_called_once()
        assert result1 == result2

    def test_function_call_with_search_tool(self):
        """Test function_call method with realistic search tool call."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {
            "content": [
                {
                    "text": '[{"text": "Found relevant FAQ entries", "section": "General", "question": "How to install?"}]'
                }
            ]
        }

        # Create a realistic tool call response object (like from OpenAI)
        class ToolCallResponse:
            def __init__(self):
                self.name = "search"
                self.arguments = '{"query": "how to install kafka"}'
                self.call_id = "call_search_123"

        response = ToolCallResponse()

        mcp_tools = MCPTools(mock_client)
        result = mcp_tools.function_call(response)

        # Should call client with parsed arguments
        mock_client.call_tool.assert_called_once_with(
            "search", {"query": "how to install kafka"}
        )

        # Should return properly formatted response
        expected = {
            "type": "function_call_output",
            "call_id": "call_search_123",
            "output": '[{"text": "Found relevant FAQ entries", "section": "General", "question": "How to install?"}]',
        }
        assert result == expected

    def test_function_call_with_add_entry_tool(self):
        """Test function_call with add_entry tool (realistic example)."""
        mock_client = Mock()
        mock_client.call_tool.return_value = {
            "content": [{"text": "Entry added successfully"}]
        }

        class ToolCallResponse:
            def __init__(self):
                self.name = "add_entry"
                self.arguments = '{"question": "How to succeed?", "answer": "Follow the course materials and practice"}'
                self.call_id = "call_add_456"

        response = ToolCallResponse()

        mcp_tools = MCPTools(mock_client)
        result = mcp_tools.function_call(response)

        expected_args = {
            "question": "How to succeed?",
            "answer": "Follow the course materials and practice",
        }
        mock_client.call_tool.assert_called_once_with("add_entry", expected_args)

        assert result["output"] == "Entry added successfully"
        assert result["call_id"] == "call_add_456"
        assert result["type"] == "function_call_output"
