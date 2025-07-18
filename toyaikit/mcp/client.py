from typing import Dict, Any, List, Optional
from toyaikit.mcp.transport import MCPTransport


class MCPClient:
    def __init__(
        self,
        transport: MCPTransport,
        client_name: str = "toyaikit",
        client_version: str = "0.0.1",
    ):
        self.transport = transport
        self.request_id = 0
        self.available_tools = {}
        self.is_initialized = False

        self.client_name = client_name
        self.client_version = client_version

    def start_server(self):
        self.transport.start()

    def stop_server(self):
        self.transport.stop()

    def _get_next_request_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }

        if params:
            notification["params"] = params

        self.transport.send(notification)

    def _send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": method,
        }

        if params:
            request["params"] = params

        self.transport.send(request)

        response = self.transport.receive()
        if "error" in response:
            raise Exception(f"Server error: {response['error']}")

        return response.get("result", {})

    def initialize(self) -> Dict[str, Any]:
        print("Sending initialize request...")
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {
                    "listChanged": True
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": self.client_name,
                "version": self.client_version
            },
        }

        result = self._send_request("initialize", params)

        print(f"Initialize response: {result}")
        return result

    def initialized(self):
        print("Sending initialized notification...")
        self._send_notification("notifications/initialized")
        self.is_initialized = True
        print("Handshake completed successfully")

    def full_initialize(self):
        self.start_server()
        self.initialize()
        self.initialized()
        self.get_tools()

    def get_tools(self) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            raise RuntimeError(
                "Client not initialized. Call initialize() and initialized() first."
            )
        print("Retrieving available tools...")
        result = self._send_request("tools/list")
        tools = result.get("tools", [])
        self.available_tools = {tool["name"]: tool for tool in tools}
        print(f"Available tools: {list(self.available_tools.keys())}")
        return tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if not self.is_initialized:
            raise RuntimeError(
                "Client not initialized. Call initialize() and initialized() first."
            )
        if tool_name not in self.available_tools:
            raise ValueError(
                f"Tool '{tool_name}' not available. Available tools: {list(self.available_tools.keys())}"
            )
        print(f"Calling tool '{tool_name}' with arguments: {arguments}")

        params = {"name": tool_name, "arguments": arguments}

        result = self._send_request("tools/call", params)
        return result

    def list_available_tools(self):
        if not self.available_tools:
            print("No tools available. Call get_tools() first.")
            return
        print("\nAvailable Tools:")
        print("-" * 50)
        for name, tool in self.available_tools.items():
            print(f"Name: {name}")
            print(f"Description: {tool.get('description', 'No description')}")
            input_schema = tool.get("inputSchema", {})
            if input_schema.get("properties"):
                print("Parameters:")
                for param_name, param_info in input_schema["properties"].items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "No description")
                    print(f"  - {param_name} ({param_type}): {param_desc}")
            print("-" * 50)
