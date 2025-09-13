from unittest.mock import Mock, patch

import pytest

from toyaikit.mcp.client import MCPClient
from toyaikit.mcp.transport import MCPTransport


class TestMCPClient:
    def test_initialization(self):
        """Test MCPClient initialization with default parameters."""
        mock_transport = Mock(spec=MCPTransport)

        client = MCPClient(mock_transport)

        assert client.transport == mock_transport
        assert client.request_id == 0
        assert client.available_tools == {}
        assert not client.is_initialized
        assert client.client_name == "toyaikit"
        assert client.client_version == "0.0.1"

    def test_initialization_with_custom_params(self):
        """Test MCPClient initialization with custom parameters."""
        mock_transport = Mock(spec=MCPTransport)

        client = MCPClient(
            mock_transport, client_name="test_client", client_version="1.0.0"
        )

        assert client.client_name == "test_client"
        assert client.client_version == "1.0.0"

    def test_start_server(self):
        """Test starting the MCP server."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        client.start_server()

        mock_transport.start.assert_called_once()

    def test_stop_server(self):
        """Test stopping the MCP server."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        client.stop_server()

        mock_transport.stop.assert_called_once()

    def test_get_next_request_id(self):
        """Test request ID generation."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        assert client._get_next_request_id() == 1
        assert client._get_next_request_id() == 2
        assert client._get_next_request_id() == 3

    def test_send_notification_without_params(self):
        """Test sending notification without parameters."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        client._send_notification("test/notification")

        expected_notification = {
            "jsonrpc": "2.0",
            "method": "test/notification",
        }
        mock_transport.send.assert_called_once_with(expected_notification)

    def test_send_notification_with_params(self):
        """Test sending notification with parameters."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        params = {"test": "value"}
        client._send_notification("test/notification", params)

        expected_notification = {
            "jsonrpc": "2.0",
            "method": "test/notification",
            "params": params,
        }
        mock_transport.send.assert_called_once_with(expected_notification)

    def test_send_request_without_params(self):
        """Test sending request without parameters."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"success": True},
        }

        client = MCPClient(mock_transport)

        result = client._send_request("test/method")

        expected_request = {"jsonrpc": "2.0", "id": 1, "method": "test/method"}
        mock_transport.send.assert_called_once_with(expected_request)
        assert result == {"success": True}

    def test_send_request_with_params(self):
        """Test sending request with parameters."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"data": "response"},
        }

        client = MCPClient(mock_transport)

        params = {"query": "test"}
        result = client._send_request("test/method", params)

        expected_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test/method",
            "params": params,
        }
        mock_transport.send.assert_called_once_with(expected_request)
        assert result == {"data": "response"}

    def test_send_request_with_server_error(self):
        """Test handling server error in request."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }

        client = MCPClient(mock_transport)

        with pytest.raises(Exception, match="Server error"):
            client._send_request("invalid/method")

    def test_initialize_success(self):
        """Test successful initialization."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
            },
        }

        client = MCPClient(mock_transport)

        result = client.initialize()

        # Verify the initialize request was sent correctly
        expected_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": True}, "sampling": {}},
            "clientInfo": {"name": "toyaikit", "version": "0.0.1"},
        }
        expected_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": expected_params,
        }
        mock_transport.send.assert_called_once_with(expected_request)

        # Verify the result
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "test-server"

    def test_initialized_notification(self):
        """Test sending initialized notification."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        client.initialized()

        expected_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        mock_transport.send.assert_called_once_with(expected_notification)
        assert client.is_initialized

    def test_get_tools_success(self):
        """Test successful tool retrieval."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
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
                    },
                    {
                        "name": "add_entry",
                        "description": "Add entry to FAQ",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                            },
                            "required": ["question", "answer"],
                        },
                    },
                ]
            },
        }

        client = MCPClient(mock_transport)
        client.is_initialized = True  # Set manually for test

        tools = client.get_tools()

        # Verify tools/list request was sent
        expected_request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        mock_transport.send.assert_called_once_with(expected_request)

        # Verify tools were parsed correctly
        assert len(tools) == 2
        assert tools[0]["name"] == "search"
        assert tools[1]["name"] == "add_entry"

        # Verify available_tools dict was populated
        assert "search" in client.available_tools
        assert "add_entry" in client.available_tools

    def test_get_tools_not_initialized(self):
        """Test get_tools fails when not initialized."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        with pytest.raises(RuntimeError, match="Client not initialized"):
            client.get_tools()

    def test_call_tool_success(self):
        """Test successful tool call."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "Search results here"}],
                "isError": False,
            },
        }

        client = MCPClient(mock_transport)
        client.is_initialized = True
        client.available_tools = {
            "search": {"name": "search", "description": "Search tool"}
        }

        result = client.call_tool("search", {"query": "how to install kafka"})

        # Verify tools/call request was sent correctly
        expected_params = {
            "name": "search",
            "arguments": {"query": "how to install kafka"},
        }
        expected_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": expected_params,
        }
        mock_transport.send.assert_called_once_with(expected_request)

        # Verify result
        assert result["content"][0]["text"] == "Search results here"

    def test_call_tool_not_initialized(self):
        """Test call_tool fails when not initialized."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        with pytest.raises(RuntimeError, match="Client not initialized"):
            client.call_tool("search", {"query": "test"})

    def test_call_tool_not_available(self):
        """Test call_tool fails when tool not available."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)
        client.is_initialized = True
        client.available_tools = {"other_tool": {}}

        with pytest.raises(ValueError, match="Tool 'search' not available"):
            client.call_tool("search", {"query": "test"})

    @patch("toyaikit.mcp.client.time.sleep")
    def test_full_initialize_success(self, mock_sleep):
        """Test full initialization sequence."""
        mock_transport = Mock(spec=MCPTransport)

        # Mock responses for initialize and get_tools
        mock_transport.receive.side_effect = [
            {  # initialize response
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "test-server", "version": "1.0.0"},
                },
            },
            {  # get_tools response
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search tool",
                            "inputSchema": {},
                        }
                    ]
                },
            },
        ]

        client = MCPClient(mock_transport)

        client.full_initialize()

        # Verify all steps were called
        mock_transport.start.assert_called_once()
        mock_sleep.assert_called_once_with(0.5)  # Default pause
        assert (
            mock_transport.send.call_count == 3
        )  # initialize, initialized notification, get_tools
        assert client.is_initialized
        assert "search" in client.available_tools

    @patch("toyaikit.mcp.client.time.sleep")
    def test_full_initialize_custom_pause(self, mock_sleep):
        """Test full initialization with custom pause."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.side_effect = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            },
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
        ]

        client = MCPClient(mock_transport)

        client.full_initialize(server_start_pause=2.0)

        mock_sleep.assert_called_once_with(2.0)

    @patch("toyaikit.mcp.client.time.sleep")
    def test_full_initialize_no_pause(self, mock_sleep):
        """Test full initialization with no pause."""
        mock_transport = Mock(spec=MCPTransport)
        mock_transport.receive.side_effect = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            },
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
        ]

        client = MCPClient(mock_transport)

        client.full_initialize(server_start_pause=0)

        mock_sleep.assert_not_called()

    def test_list_available_tools_with_tools(self):
        """Test listing available tools when tools exist."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        client.available_tools = {
            "search": {
                "name": "search",
                "description": "Search the FAQ database",
                "inputSchema": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                        }
                    }
                },
            }
        }

        # This method prints output, so we just verify it doesn't crash
        client.list_available_tools()

    def test_list_available_tools_no_tools(self):
        """Test listing available tools when no tools exist."""
        mock_transport = Mock(spec=MCPTransport)
        client = MCPClient(mock_transport)

        # This method prints output, so we just verify it doesn't crash
        client.list_available_tools()

    def test_realistic_faq_workflow(self):
        """Test realistic FAQ workflow with search and add_entry tools."""
        mock_transport = Mock(spec=MCPTransport)

        # Mock the complete workflow responses
        mock_transport.receive.side_effect = [
            {  # initialize
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "FAQ Server", "version": "1.0.0"},
                },
            },
            {  # get_tools
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search the FAQ database for entries matching the given query.",
                            "inputSchema": {
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "add_entry",
                            "description": "Add a new entry to the FAQ database.",
                            "inputSchema": {
                                "properties": {
                                    "question": {"type": "string"},
                                    "answer": {"type": "string"},
                                },
                                "required": ["question", "answer"],
                            },
                        },
                    ]
                },
            },
            {  # search call
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": '[{"question": "How to install Kafka?", "answer": "Use Docker..."}]',
                        }
                    ],
                    "isError": False,
                },
            },
            {  # add_entry call
                "jsonrpc": "2.0",
                "id": 4,
                "result": {
                    "content": [{"type": "text", "text": "Entry added successfully"}],
                    "isError": False,
                },
            },
        ]

        client = MCPClient(mock_transport)

        # Initialize
        client.start_server()
        client.initialize()
        client.initialized()
        tools = client.get_tools()

        # Verify tools are available
        assert len(tools) == 2
        assert "search" in client.available_tools
        assert "add_entry" in client.available_tools

        # Test search
        search_result = client.call_tool("search", {"query": "how to install kafka"})
        assert "content" in search_result

        # Test add_entry
        add_result = client.call_tool(
            "add_entry",
            {
                "question": "How to succeed in Module 1?",
                "answer": "Follow the course materials and practice",
            },
        )
        assert "content" in add_result
