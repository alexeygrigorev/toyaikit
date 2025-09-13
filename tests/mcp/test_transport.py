import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from toyaikit.mcp.transport import MCPTransport, SubprocessMCPTransport


class TestMCPTransport:
    def test_base_class_methods_not_implemented(self):
        """Test that base class methods raise NotImplementedError."""
        transport = MCPTransport()

        with pytest.raises(NotImplementedError):
            transport.start()

        with pytest.raises(NotImplementedError):
            transport.stop()

        with pytest.raises(NotImplementedError):
            transport.send({})

        with pytest.raises(NotImplementedError):
            transport.receive()


class TestSubprocessMCPTransport:
    def test_initialization(self):
        """Test transport initialization with command and workdir."""
        command = ["python", "-m", "server"]
        workdir = "/test/dir"

        transport = SubprocessMCPTransport(command, workdir)

        assert transport.server_command == command
        assert transport.workdir == workdir
        assert transport.process is None

    def test_initialization_without_workdir(self):
        """Test transport initialization without workdir."""
        command = ["python", "-m", "server"]

        transport = SubprocessMCPTransport(command)

        assert transport.server_command == command
        assert transport.workdir is None
        assert transport.process is None

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    @patch("toyaikit.mcp.transport.os.environ")
    def test_start_subprocess(self, mock_environ, mock_popen):
        """Test starting subprocess with proper configuration."""
        mock_environ.copy.return_value = {}
        mock_process = Mock()
        mock_popen.return_value = mock_process

        command = ["python", "-m", "server"]
        transport = SubprocessMCPTransport(command, "/test/dir")

        transport.start()

        # Verify subprocess.Popen was called with correct parameters
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args

        assert call_args[0][0] == command  # First positional arg
        assert call_args[1]["stdin"] == subprocess.PIPE
        assert call_args[1]["stdout"] == subprocess.PIPE
        assert call_args[1]["stderr"] == subprocess.PIPE
        assert call_args[1]["text"] is True
        assert call_args[1]["bufsize"] == 0
        assert call_args[1]["cwd"] == "/test/dir"
        assert call_args[1]["encoding"] == "utf-8"
        assert call_args[1]["errors"] == "replace"

        # Verify environment variables were set
        env = call_args[1]["env"]
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONUNBUFFERED"] == "1"

        assert transport.process == mock_process

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_stop_process(self, mock_popen):
        """Test stopping the subprocess."""
        mock_process = Mock()
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()
        transport.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()

    def test_stop_without_process(self):
        """Test stopping when no process exists."""
        transport = SubprocessMCPTransport(["python", "-m", "server"])

        # Should not raise an error
        transport.stop()

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_send_data(self, mock_popen):
        """Test sending JSON data to subprocess."""
        mock_process = Mock()
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()

        test_data = {"method": "test", "params": {"value": 42}}
        transport.send(test_data)

        expected_json = json.dumps(test_data, ensure_ascii=False) + "\n"
        mock_process.stdin.write.assert_called_once_with(expected_json)
        mock_process.stdin.flush.assert_called_once()

    def test_send_without_process(self):
        """Test sending data when process is not started."""
        transport = SubprocessMCPTransport(["python", "-m", "server"])

        with pytest.raises(RuntimeError, match="Server not started"):
            transport.send({"test": "data"})

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_send_unicode_data(self, mock_popen):
        """Test sending Unicode data."""
        mock_process = Mock()
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()

        # Test with Unicode characters
        test_data = {"message": "Hello 世界 🌍"}
        transport.send(test_data)

        expected_json = json.dumps(test_data, ensure_ascii=False) + "\n"
        mock_process.stdin.write.assert_called_once_with(expected_json)

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_send_unicode_error(self, mock_popen):
        """Test handling Unicode encoding errors during send."""
        mock_process = Mock()
        mock_process.stdin.write.side_effect = UnicodeEncodeError(
            "utf-8", "test string", 0, 1, "invalid start byte"
        )
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()

        with pytest.raises(RuntimeError, match="Unicode encoding error"):
            transport.send({"test": "data"})

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_receive_data(self, mock_popen):
        """Test receiving JSON data from subprocess."""
        mock_process = Mock()
        test_response = {"result": "success", "value": 123}
        mock_process.stdout.readline.return_value = json.dumps(test_response) + "\n"
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()

        result = transport.receive()

        assert result == test_response
        mock_process.stdout.readline.assert_called_once()

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_receive_empty_response(self, mock_popen):
        """Test handling empty response from subprocess."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = ""
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()

        with pytest.raises(RuntimeError, match="No response from server"):
            transport.receive()

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_receive_invalid_json(self, mock_popen):
        """Test handling invalid JSON response."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = "invalid json"
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()

        with pytest.raises(RuntimeError, match="JSON decode error"):
            transport.receive()

    def test_receive_without_process(self):
        """Test receiving data when process is not started."""
        transport = SubprocessMCPTransport(["python", "-m", "server"])

        with pytest.raises(RuntimeError, match="Server not started"):
            transport.receive()

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_receive_unicode_decode_error(self, mock_popen):
        """Test handling Unicode decoding errors during receive."""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = UnicodeDecodeError(
            "utf-8", b"invalid bytes", 0, 1, "invalid start byte"
        )
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])
        transport.start()

        with pytest.raises(RuntimeError, match="Unicode decoding error"):
            transport.receive()

    @patch("toyaikit.mcp.transport.subprocess.Popen")
    def test_full_lifecycle(self, mock_popen):
        """Test complete start-send-receive-stop lifecycle."""
        mock_process = Mock()
        mock_process.stdout.readline.return_value = '{"status": "ok"}\n'
        mock_popen.return_value = mock_process

        transport = SubprocessMCPTransport(["python", "-m", "server"])

        # Start
        transport.start()
        assert transport.process == mock_process

        # Send
        transport.send({"command": "ping"})
        mock_process.stdin.write.assert_called()

        # Receive
        result = transport.receive()
        assert result == {"status": "ok"}

        # Stop
        transport.stop()
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
