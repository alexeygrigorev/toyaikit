import json
import os
import subprocess
from typing import Any, Dict, List


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
    def __init__(self, server_command: List[str], workdir: str = None):
        self.server_command = server_command
        self.workdir = workdir
        self.process = None

    def start(self):
        # Ensure proper unicode support with UTF-8 encoding
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            cwd=self.workdir,
            env=env,
            encoding="utf-8",
            errors="replace",
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
        try:
            data_str = json.dumps(data, ensure_ascii=False) + "\n"
            self.process.stdin.write(data_str)
            self.process.stdin.flush()
        except UnicodeEncodeError as e:
            raise RuntimeError(f"Unicode encoding error: {e}")

    def receive(self) -> Dict[str, Any]:
        if not self.process:
            raise RuntimeError("Server not started")
        try:
            response_str = self.process.stdout.readline().strip()
            if not response_str:
                raise RuntimeError("No response from server")
            return json.loads(response_str)
        except UnicodeDecodeError as e:
            raise RuntimeError(f"Unicode decoding error: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON decode error: {e}")
