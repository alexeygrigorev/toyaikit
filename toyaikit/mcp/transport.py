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

    def is_alive(self) -> bool:
        """Check if the subprocess is still running."""
        return self.process is not None and self.process.poll() is None

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
            try:
                if self.process.poll() is None:  # Process is still running
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5.0)  # Wait up to 5 seconds
                    except subprocess.TimeoutExpired:
                        self.process.kill()  # Force kill if terminate doesn't work
                        self.process.wait()
                else:
                    # Process already terminated, just clean up
                    pass
            except OSError:
                # Process might already be dead or inaccessible
                pass
            finally:
                # Close pipes to prevent resource leaks
                if self.process.stdin:
                    try:
                        self.process.stdin.close()
                    except (OSError, BrokenPipeError):
                        pass
                if self.process.stdout:
                    try:
                        self.process.stdout.close()
                    except OSError:
                        pass
                if self.process.stderr:
                    try:
                        self.process.stderr.close()
                    except OSError:
                        pass
                print("Server stopped")

    def send(self, data: Dict[str, Any]):
        if not self.process:
            raise RuntimeError("Server not started")
        if not self.is_alive():
            raise RuntimeError("Server process has terminated")
        try:
            data_str = json.dumps(data, ensure_ascii=False) + "\n"
            self.process.stdin.write(data_str)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, UnicodeError) as e:
            if isinstance(e, BrokenPipeError):
                raise RuntimeError("Server process has terminated (broken pipe)")
            elif isinstance(e, UnicodeError):
                raise RuntimeError(f"Unicode encoding error: {e}")
            else:
                raise RuntimeError(f"Communication error: {e}")

    def receive(self) -> Dict[str, Any]:
        if not self.process:
            raise RuntimeError("Server not started")
        if not self.is_alive():
            raise RuntimeError("Server process has terminated")
        try:
            response_str = self.process.stdout.readline().strip()
            if not response_str:
                raise RuntimeError("No response from server")
            return json.loads(response_str)
        except (UnicodeDecodeError, json.JSONDecodeError, OSError) as e:
            if isinstance(e, UnicodeDecodeError):
                raise RuntimeError(f"Unicode decoding error: {e}")
            elif isinstance(e, json.JSONDecodeError):
                raise RuntimeError(f"JSON decode error: {e}")
            else:
                raise RuntimeError(f"Communication error: {e}")
