import pytest
from unittest.mock import MagicMock, patch
from toyaikit.chat.chat import LLMClient, OpenAIClient, ChatAssistant

# Test LLMClient base class
def test_llmclient_send_request_not_implemented():
    client = LLMClient()
    with pytest.raises(NotImplementedError):
        client.send_request([{"role": "user", "content": "hi"}])

# Test OpenAIClient with mocked OpenAI and Tools
@patch('toyaikit.chat.chat.OpenAI')
def test_openaiclient_send_request(mock_openai):
    mock_tools = MagicMock()
    mock_tools.get_tools.return_value = ["tool1"]
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_client.responses.create.return_value = mock_response

    client = OpenAIClient(tools=mock_tools, model="gpt-4o-mini")
    chat_messages = [
        {"role": "user", "content": "hi"}
    ]
    result = client.send_request(chat_messages)
    mock_client.responses.create.assert_called_once_with(
        model="gpt-4o-mini",
        input=chat_messages,
        tools=["tool1"]
    )
    assert result == mock_response

# Test ChatAssistant run loop (simulate one user input and response)
def test_chatassistant_run_one_cycle(monkeypatch):
    mock_tools = MagicMock()
    mock_tools.function_call.return_value = {"role": "function", "content": "result"}
    mock_interface = MagicMock()
    # Simulate user input: first call returns 'hello', second call returns 'stop'
    mock_interface.input.side_effect = ["hello", "stop"]
    mock_llm_client = MagicMock()
    # Simulate LLM response: one message
    mock_response = MagicMock()
    mock_response.output = [
        MagicMock(type="message", **{"__getitem__": lambda self, key: {"type": "message", "content": "hi"}[key]})
    ]
    mock_llm_client.send_request.return_value = mock_response

    assistant = ChatAssistant(
        tools=mock_tools,
        developer_prompt="You are a helpful assistant.",
        chat_interface=mock_interface,
        llm_client=mock_llm_client
    )
    # Patch display methods to avoid real output
    mock_interface.display = MagicMock()
    mock_interface.display_response = MagicMock()
    mock_interface.display_function_call = MagicMock()

    # Run the assistant (should exit after one cycle)
    assistant.run()
    # Check that input was called at least twice ("hello" and "stop")
    assert mock_interface.input.call_count >= 2
    # Check that display_response was called
    mock_interface.display_response.assert_called() 