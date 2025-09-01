import pytest
from unittest.mock import patch, MagicMock
from toyaikit.chat.interface import StdOutputInterface


class TestStdOutputInterface:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.interface = StdOutputInterface()

    def test_input_returns_stripped_input(self):
        """Test that input method returns stripped user input."""
        with patch('builtins.input', return_value="  hello world  "):
            result = self.interface.input()
            assert result == "hello world"

    def test_display_prints_message(self, capsys):
        """Test that display method prints the message to stdout."""
        test_message = "Test message"
        self.interface.display(test_message)
        captured = capsys.readouterr()
        assert captured.out == test_message + "\n"

    def test_display_function_call_formats_output(self, capsys):
        """Test that display_function_call formats and prints function call information."""
        function_name = "test_function"
        arguments = '{"param": "value"}'
        result = "success"
        
        self.interface.display_function_call(function_name, arguments, result)
        captured = capsys.readouterr()
        
        output = captured.out
        assert "--- Function Call ---" in output
        assert f"Function: {function_name}" in output
        assert f"Arguments: {arguments}" in output
        assert f"Result: {result}" in output
        assert "-------------------" in output

    def test_display_response_formats_output(self, capsys):
        """Test that display_response formats and prints the response."""
        markdown_text = "This is a **bold** response"
        
        self.interface.display_response(markdown_text)
        captured = capsys.readouterr()
        
        output = captured.out
        assert "Assistant:" in output
        assert markdown_text in output

    def test_display_reasoning_formats_output(self, capsys):
        """Test that display_reasoning formats and prints the reasoning."""
        markdown_text = "This is the reasoning behind the answer"
        
        self.interface.display_reasoning(markdown_text)
        captured = capsys.readouterr()
        
        output = captured.out
        assert "--- Reasoning ---" in output
        assert markdown_text in output
        assert "---------------" in output

    def test_all_methods_implemented(self):
        """Test that all required methods are implemented."""
        # Check that all required methods exist
        assert hasattr(self.interface, 'input')
        assert hasattr(self.interface, 'display')
        assert hasattr(self.interface, 'display_function_call')
        assert hasattr(self.interface, 'display_response')
        assert hasattr(self.interface, 'display_reasoning')
        
        # Check that all methods are callable
        assert callable(self.interface.input)
        assert callable(self.interface.display)
        assert callable(self.interface.display_function_call)
        assert callable(self.interface.display_response)
        assert callable(self.interface.display_reasoning)


