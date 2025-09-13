from unittest.mock import Mock, patch

import pytest
from openai import OpenAI
from pydantic import BaseModel

from toyaikit.llm import LLMClient, OpenAIChatCompletionsClient, OpenAIClient
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

    def test_convert_single_tool_not_function(self):
        """Test convert_single_tool raises error for non-function tool"""
        mock_client = Mock(spec=OpenAI)
        client = OpenAIChatCompletionsClient(client=mock_client)

        tool = {"type": "other", "name": "not_a_function"}

        with pytest.raises(
            TypeError, match="exceptions must derive from BaseException"
        ):
            client.convert_single_tool(tool)

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
        mock_client.chat.completions.create.return_value = mock_response

        # Create a mock BaseModel for output format
        class TestOutputFormat(BaseModel):
            field: str

        client = OpenAIChatCompletionsClient(client=mock_client)
        chat_messages = [{"role": "user", "content": "Hello"}]

        result = client.send_request(chat_messages, output_format=TestOutputFormat)

        assert result == mock_response
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini", messages=chat_messages, tools=[]
        )
