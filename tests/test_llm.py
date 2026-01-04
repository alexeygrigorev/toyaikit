from unittest.mock import Mock, patch

import pytest
from openai import OpenAI
from pydantic import BaseModel

from toyaikit.llm import LLMClient, OpenAIChatCompletionsClient, OpenAIClient, AnthropicClient
from toyaikit.tools import Tools


class TestLLMClient:
    def test_base_class_send_request_not_implemented(self):
        """Test base class raises NotImplementedError"""
        client = LLMClient()
        with pytest.raises(
            NotImplementedError, match="Subclasses must implement this method"
        ):
            client.send_request([])


class TestOpenAIClient:
    def test_initialization_with_defaults(self):
        """Test OpenAIClient initialization with default parameters"""
        with patch("toyaikit.llm.OpenAI") as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            client = OpenAIClient()

            assert client.model == "gpt-4o-mini"
            assert client.client == mock_client_instance
            assert client.extra_kwargs == {}
            mock_openai.assert_called_once()

    def test_initialization_with_custom_model(self):
        """Test OpenAIClient initialization with custom model"""
        with patch("toyaikit.llm.OpenAI") as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            client = OpenAIClient(model="gpt-4o")

            assert client.model == "gpt-4o"
            assert client.client == mock_client_instance
            assert client.extra_kwargs == {}

    def test_initialization_with_custom_client(self):
        """Test OpenAIClient initialization with provided client"""
        mock_client = Mock(spec=OpenAI)

        client = OpenAIClient(client=mock_client)

        assert client.model == "gpt-4o-mini"
        assert client.client == mock_client
        assert client.extra_kwargs == {}

    def test_initialization_with_extra_kwargs(self):
        """Test OpenAIClient initialization with extra kwargs"""
        mock_client = Mock(spec=OpenAI)
        extra_kwargs = {"temperature": 0.7, "max_tokens": 1000}

        client = OpenAIClient(client=mock_client, extra_kwargs=extra_kwargs)

        assert client.model == "gpt-4o-mini"
        assert client.client == mock_client
        assert client.extra_kwargs == extra_kwargs

    def test_send_request_without_tools(self):
        """Test send_request without tools"""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_client.responses.create.return_value = mock_response

        client = OpenAIClient(client=mock_client)
        chat_messages = [{"role": "user", "content": "Hello"}]

        result = client.send_request(chat_messages)

        assert result == mock_response
        mock_client.responses.create.assert_called_once_with(
            model="gpt-4o-mini", input=chat_messages, tools=[]
        )

    def test_send_request_with_tools(self):
        """Test send_request with tools"""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_client.responses.create.return_value = mock_response

        tools = Mock(spec=Tools)
        tools_list = [{"name": "test_tool", "description": "A test tool"}]
        tools.get_tools.return_value = tools_list

        client = OpenAIClient(client=mock_client)
        chat_messages = [{"role": "user", "content": "Hello"}]

        result = client.send_request(chat_messages, tools=tools)

        assert result == mock_response
        tools.get_tools.assert_called_once()
        mock_client.responses.create.assert_called_once_with(
            model="gpt-4o-mini", input=chat_messages, tools=tools_list
        )

    def test_send_request_with_extra_kwargs(self):
        """Test send_request passes extra kwargs"""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_client.responses.create.return_value = mock_response

        extra_kwargs = {"temperature": 0.7, "max_tokens": 1000}
        client = OpenAIClient(client=mock_client, extra_kwargs=extra_kwargs)
        chat_messages = [{"role": "user", "content": "Hello"}]

        result = client.send_request(chat_messages)

        assert result == mock_response
        mock_client.responses.create.assert_called_once_with(
            model="gpt-4o-mini",
            input=chat_messages,
            tools=[],
            temperature=0.7,
            max_tokens=1000,
        )


class TestOpenAIChatCompletionsClient:
    def test_initialization_with_defaults(self):
        """Test OpenAIChatCompletionsClient initialization with default parameters"""
        with patch("toyaikit.llm.OpenAI") as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            client = OpenAIChatCompletionsClient()

            assert client.model == "gpt-4o-mini"
            assert client.client == mock_client_instance
            mock_openai.assert_called_once()

    def test_initialization_with_custom_client(self):
        """Test OpenAIChatCompletionsClient initialization with provided client"""
        mock_client = Mock(spec=OpenAI)

        client = OpenAIChatCompletionsClient(model="gpt-4o", client=mock_client)

        assert client.model == "gpt-4o"
        assert client.client == mock_client

    def test_convert_single_tool_success(self):
        """Test convert_single_tool with valid function tool"""
        mock_client = Mock(spec=OpenAI)
        client = OpenAIChatCompletionsClient(client=mock_client)

        tool = {
            "type": "function",
            "name": "search",
            "description": "Search the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        }

        result = client.convert_single_tool(tool)

        expected = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the database",
                "parameters": {
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
        }

        assert result == expected

    def test_convert_api_tools_to_chat_functions(self):
        """Test convert_api_tools_to_chat_functions with multiple tools"""
        mock_client = Mock(spec=OpenAI)
        client = OpenAIChatCompletionsClient(client=mock_client)

        api_tools = [
            {
                "type": "function",
                "name": "search",
                "description": "Search the database",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "add_entry",
                "description": "Add an entry",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

        result = client.convert_api_tools_to_chat_functions(api_tools)

        expected = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the database",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_entry",
                    "description": "Add an entry",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        assert result == expected

    def test_convert_api_tools_to_chat_functions_empty_list(self):
        """Test convert_api_tools_to_chat_functions with empty list"""
        mock_client = Mock(spec=OpenAI)
        client = OpenAIChatCompletionsClient(client=mock_client)

        result = client.convert_api_tools_to_chat_functions([])

        assert result == []

    def test_send_request_without_tools_and_output_format(self):
        """Test send_request without tools or output format"""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIChatCompletionsClient(client=mock_client)
        chat_messages = [{"role": "user", "content": "Hello"}]

        result = client.send_request(chat_messages)

        assert result == mock_response
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini", messages=chat_messages, tools=[]
        )

    def test_send_request_with_tools(self):
        """Test send_request with tools"""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        tools = Mock(spec=Tools)
        api_tools = [
            {
                "type": "function",
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        tools.get_tools.return_value = api_tools

        client = OpenAIChatCompletionsClient(client=mock_client)
        chat_messages = [{"role": "user", "content": "Hello"}]

        result = client.send_request(chat_messages, tools=tools)

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        assert result == mock_response
        tools.get_tools.assert_called_once()
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini", messages=chat_messages, tools=expected_tools
        )

    def test_send_request_with_output_format(self):
        """Test send_request with output_format parameter"""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_client.chat.completions.parse.return_value = mock_response

        # Create a mock BaseModel for output format
        class TestOutputFormat(BaseModel):
            field: str

        client = OpenAIChatCompletionsClient(client=mock_client)
        chat_messages = [{"role": "user", "content": "Hello"}]

        result = client.send_request(chat_messages, output_format=TestOutputFormat)

        assert result == mock_response
        mock_client.chat.completions.parse.assert_called_once_with(
            model="gpt-4o-mini",
            messages=chat_messages,
            tools=[],
            response_format=TestOutputFormat,
        )

    def test_convert_api_tools_to_chat_functions_strict(self):
        """When strict=True, tools include strict flag on function."""
        mock_client = Mock(spec=OpenAI)
        client = OpenAIChatCompletionsClient(client=mock_client)

        api_tools = [
            {
                "type": "function",
                "name": "calc",
                "description": "Calculate",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        result = client.convert_api_tools_to_chat_functions(api_tools, strict=True)
        assert result[0]["function"]["strict"] is True

    def test_send_request_with_output_format_adds_strict_on_tools(self):
        """parse() should receive tools with strict=True when tools are provided."""
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_client.chat.completions.parse.return_value = mock_response

        # Create a mock BaseModel for output format
        class TestOutputFormat(BaseModel):
            field: str

        # One API tool returned by Tools.get_tools()
        tools = Mock(spec=Tools)
        api_tools = [
            {
                "type": "function",
                "name": "calc",
                "description": "Calculate",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        tools.get_tools.return_value = api_tools

        client = OpenAIChatCompletionsClient(client=mock_client)
        chat_messages = [{"role": "user", "content": "Hello"}]

        _ = client.send_request(chat_messages, tools=tools, output_format=TestOutputFormat)

        # Capture call and verify strict flag is present on tool
        _, kwargs = mock_client.chat.completions.parse.call_args
        sent_tools = kwargs["tools"]
        assert isinstance(sent_tools, list) and len(sent_tools) == 1
        assert sent_tools[0]["function"]["strict"] is True


class TestAnthropicClient:
    def test_initialization_with_defaults(self):
        """Test AnthropicClient initialization with default parameters"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client_instance = Mock()
            mock_anthropic.return_value = mock_client_instance

            client = AnthropicClient()

            assert client.model == "claude-sonnet-4-5-20250514"
            assert client.client == mock_client_instance
            assert client.extra_kwargs == {}
            mock_anthropic.assert_called_once_with()

    def test_initialization_with_custom_model(self):
        """Test AnthropicClient initialization with custom model"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client_instance = Mock()
            mock_anthropic.return_value = mock_client_instance

            client = AnthropicClient(model="claude-haiku-4-20250514")

            assert client.model == "claude-haiku-4-20250514"
            assert client.client == mock_client_instance

    def test_initialization_with_api_key(self):
        """Test AnthropicClient initialization with API key"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client_instance = Mock()
            mock_anthropic.return_value = mock_client_instance

            client = AnthropicClient(api_key="test-api-key")

            mock_anthropic.assert_called_once_with(api_key="test-api-key")

    def test_initialization_with_base_url(self):
        """Test AnthropicClient initialization with base URL (for compatible APIs)"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client_instance = Mock()
            mock_anthropic.return_value = mock_client_instance

            client = AnthropicClient(base_url="https://api.example.com")

            mock_anthropic.assert_called_once_with(base_url="https://api.example.com")

    def test_initialization_with_extra_kwargs(self):
        """Test AnthropicClient initialization with extra kwargs"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client_instance = Mock()
            mock_anthropic.return_value = mock_client_instance

            extra_kwargs = {"temperature": 0.7, "max_tokens": 1000}
            client = AnthropicClient(extra_kwargs=extra_kwargs)

            assert client.extra_kwargs == extra_kwargs

    def test_import_error_when_anthropic_not_installed(self):
        """Test ImportError is raised when anthropic package is not available"""
        # Patch the import at the module level before client initialization
        import sys
        anthropic_module = sys.modules.get('anthropic')
        try:
            # Remove anthropic from sys.modules to trigger ImportError
            if 'anthropic' in sys.modules:
                del sys.modules['anthropic']

            # Mock the import to fail
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == 'anthropic':
                    raise ImportError("No module named 'anthropic'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, '__import__', side_effect=mock_import):
                with pytest.raises(ImportError, match="Please run 'pip install anthropic'"):
                    AnthropicClient()
        finally:
            # Restore the original module if it existed
            if anthropic_module is not None:
                sys.modules['anthropic'] = anthropic_module

    def test_convert_openai_tool_to_anthropic(self):
        """Test converting OpenAI tool format to Anthropic format"""
        with patch("anthropic.Anthropic"):
            client = AnthropicClient()

            openai_tool = {
                "type": "function",
                "name": "search",
                "description": "Search the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }

            result = client.convert_openai_tool_to_anthropic(openai_tool)

            expected = {
                "name": "search",
                "description": "Search the database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }

            assert result == expected

    def test_send_request_without_tools(self):
        """Test send_request without tools"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Hello")]
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            client = AnthropicClient()
            chat_messages = [{"role": "user", "content": "Hello"}]

            result = client.send_request(chat_messages)

            assert result == mock_response

            # Verify the call was made with correct arguments
            call_args = mock_client.messages.create.call_args
            kwargs = call_args[1]
            assert kwargs["model"] == "claude-sonnet-4-5-20250514"
            assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_send_request_with_system_message(self):
        """Test send_request with system message"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Hello")]
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            client = AnthropicClient()
            chat_messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"}
            ]

            result = client.send_request(chat_messages)

            call_args = mock_client.messages.create.call_args
            kwargs = call_args[1]
            assert kwargs["system"] == "You are a helpful assistant"
            assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_send_request_with_tools(self):
        """Test send_request with tools"""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.messages.create.return_value = mock_response

        tools = Mock(spec=Tools)
        tools_list = [
            {
                "type": "function",
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        tools.get_tools.return_value = tools_list

        with patch("anthropic.Anthropic", return_value=mock_client):
            client = AnthropicClient()
            chat_messages = [{"role": "user", "content": "Hello"}]

            result = client.send_request(chat_messages, tools=tools)

            call_args = mock_client.messages.create.call_args
            kwargs = call_args[1]

            # Tools should be converted to Anthropic format
            expected_tools = [
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ]
            assert kwargs["tools"] == expected_tools

    def test_send_request_with_extra_kwargs(self):
        """Test send_request passes extra kwargs"""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            extra_kwargs = {"temperature": 0.7, "max_tokens": 1000}
            client = AnthropicClient(extra_kwargs=extra_kwargs)
            chat_messages = [{"role": "user", "content": "Hello"}]

            result = client.send_request(chat_messages)

            call_args = mock_client.messages.create.call_args
            kwargs = call_args[1]
            assert kwargs["temperature"] == 0.7
            assert kwargs["max_tokens"] == 1000

    def test_send_request_with_output_format(self):
        """Test send_request with output_format (structured output)"""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.messages.create.return_value = mock_response

        class TestOutputFormat(BaseModel):
            field: str

        with patch("anthropic.Anthropic", return_value=mock_client):
            client = AnthropicClient()
            chat_messages = [{"role": "user", "content": "Hello"}]

            result = client.send_request(chat_messages, output_format=TestOutputFormat)

            call_args = mock_client.messages.create.call_args
            kwargs = call_args[1]
            assert "response_format" in kwargs
            assert kwargs["response_format"]["type"] == "json_schema"
