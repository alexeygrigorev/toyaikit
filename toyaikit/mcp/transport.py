import json
import subprocess
from typing import Dict, Any, List

class MCPTransport:
    
    def start(self):
        raise NotImplementedError("Subclasses must implement this method")

    def stop(self):
        raise NotImplementedError("Subclasses must implement this method")

    def send(self, data: Dict[str, Any]):
        raise NotImplementedError("Subclasses must implement this method")

    def receive(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")


class SubprocessMCPTransport(MCPTransport):
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        print(f"Started server with command: {' '.join(self.server_command)}")

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server stopped")

    def send(self, data: Dict[str, Any]):
        if not self.process:
            raise RuntimeError("Server not started")
        data_str = json.dumps(data) + "\n"
        self.process.stdin.write(data_str)
        self.process.stdin.flush()

    def receive(self) -> Dict[str, Any]:
        if not self.process:
            raise RuntimeError("Server not started")
        response_str = self.process.stdout.readline().strip()
        if not response_str:
            raise RuntimeError("No response from server")
        return json.loads(response_str) 