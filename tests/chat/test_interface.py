from unittest.mock import patch

from toyaikit.chat.interface import (
    IPythonChatInterface,
    StdOutputInterface,
    shorten,
)


class TestStdOutputInterface:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.interface = StdOutputInterface()

    def test_input_returns_stripped_input(self):
        """Test that input method returns stripped user input."""
        with patch("builtins.input", return_value="  hello world  "):
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
        assert hasattr(self.interface, "input")
        assert hasattr(self.interface, "display")
        assert hasattr(self.interface, "display_function_call")
        assert hasattr(self.interface, "display_response")
        assert hasattr(self.interface, "display_reasoning")

        # Check that all methods are callable
        assert callable(self.interface.input)
        assert callable(self.interface.display)
        assert callable(self.interface.display_function_call)
        assert callable(self.interface.display_response)
        assert callable(self.interface.display_reasoning)


class TestShortenFunction:
    def test_shorten_text_under_max_length(self):
        """Test that text under max length is returned unchanged."""
        text = "Short text"
        result = shorten(text, max_length=50)
        assert result == text

    def test_shorten_text_at_max_length(self):
        """Test that text exactly at max length is returned unchanged."""
        text = "a" * 50
        result = shorten(text, max_length=50)
        assert result == text

    def test_shorten_text_over_max_length(self):
        """Test that text over max length is shortened with ellipsis."""
        text = "This is a very long text that should be shortened"
        result = shorten(text, max_length=20)
        expected = "This is a very lo..."
        assert result == expected
        assert len(result) == 20

    def test_shorten_with_default_max_length(self):
        """Test that default max_length=50 is used when not specified."""
        long_text = "a" * 60
        result = shorten(long_text)
        expected = "a" * 47 + "..."
        assert result == expected
        assert len(result) == 50


class TestIPythonChatInterface:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.interface = IPythonChatInterface()

    def test_input_returns_stripped_input(self):
        """Test that input method returns stripped user input."""
        with patch("builtins.input", return_value="  hello world  "):
            result = self.interface.input()
            assert result == "hello world"

    def test_display_prints_message(self, capsys):
        """Test that display method prints the message to stdout."""
        test_message = "Test message"
        self.interface.display(test_message)
        captured = capsys.readouterr()
        assert captured.out == test_message + "\n"

    @patch("toyaikit.chat.interface.ip_display")
    def test_display_function_call_creates_html(self, mock_display):
        """Test that display_function_call creates HTML with collapsible details."""
        function_name = "test_function"
        arguments = '{"param": "value"}'
        result = "success"

        self.interface.display_function_call(function_name, arguments, result)

        # Verify ip_display was called once
        mock_display.assert_called_once()

        # Get the HTML object that was passed to ip_display
        html_call = mock_display.call_args[0][0]
        html_content = html_call.data

        # Verify the HTML contains expected elements
        assert "<details>" in html_content
        assert f"<tt>{function_name}" in html_content
        assert arguments in html_content
        assert result in html_content
        assert "</details>" in html_content

    @patch("toyaikit.chat.interface.ip_display")
    def test_display_reasoning_creates_html(self, mock_display):
        """Test that display_reasoning creates HTML with collapsible reasoning."""
        markdown_text = "This is **bold** reasoning"

        self.interface.display_reasoning(markdown_text)

        # Verify ip_display was called once
        mock_display.assert_called_once()

        # Get the HTML object that was passed to ip_display
        html_call = mock_display.call_args[0][0]
        html_content = html_call.data

        # Verify the HTML contains expected elements
        assert "<details>" in html_content
        assert "<summary>Reasoning</summary>" in html_content
        assert (
            "<strong>bold</strong>" in html_content
        )  # mistune should convert **bold** to <strong>bold</strong>

    @patch("toyaikit.chat.interface.ip_display")
    def test_display_response_creates_html(self, mock_display):
        """Test that display_response creates HTML with formatted response."""
        markdown_text = "This is a **bold** response"

        self.interface.display_response(markdown_text)

        # Verify ip_display was called once
        mock_display.assert_called_once()

        # Get the HTML object that was passed to ip_display
        html_call = mock_display.call_args[0][0]
        html_content = html_call.data

        # Verify the HTML contains expected elements
        assert "<b>Assistant:</b>" in html_content
        assert (
            "<strong>bold</strong>" in html_content
        )  # mistune should convert **bold** to <strong>bold</strong>
