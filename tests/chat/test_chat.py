from unittest.mock import MagicMock, patch, call
from toyaikit.chat.chat import OpenAIClient, ChatAssistant


class AttrDict(dict):
    def __getattr__(self, item):
        return self[item]
    def __setattr__(self, key, value):
        self[key] = value


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


def test_chatassistant_run_one_cycle(monkeypatch):
    mock_tools = MagicMock()
    mock_tools.function_call.return_value = {"role": "function", "content": "result"}
    
    mock_interface = MagicMock()
    # Simulate user input: first call returns 'hello', second call returns 'stop'
    mock_interface.input.side_effect = ["hello", "stop"]

    mock_interface.display = MagicMock()
    mock_interface.display_response = MagicMock()
    mock_interface.display_function_call = MagicMock()


    mock_llm_client = MagicMock()
    
    # Simulate LLM response: one message
    message = AttrDict({"type": "message", "content": "hi"})
    response = AttrDict({'output': [message]})

    mock_llm_client.send_request.return_value = response

    assistant = ChatAssistant(
        tools=mock_tools,
        developer_prompt="You are a helpful assistant.",
        chat_interface=mock_interface,
        llm_client=mock_llm_client
    )

    # Run the assistant (should exit after one cycle)
    assistant.run()

    assert mock_interface.input.call_count >= 2
    mock_interface.display_response.assert_called()



def test_chatassistant_function_call_flow_with_fakes():
    mock_tools = MagicMock()
    call1 = {"role": "function", "content": "result_func1"}
    call2 = {"role": "function", "content": "result_func2"}
    mock_tools.function_call.side_effect = [call1, call2]

    mock_interface = MagicMock()
    mock_interface.input.side_effect = ['ask', 'stop']

    mock_llm_client = MagicMock()

    function_call1 = AttrDict({'type': 'function_call', 'name': 'func1', 'arguments': '{}'})
    function_call2 = AttrDict({'type': 'function_call', 'name': 'func2', 'arguments': '{}'})
    message = AttrDict({'type': 'message', 'content': [{'text': 'Here is your answer.'}]})
    
    mock_llm_client.send_request.side_effect = [
        MagicMock(output=[function_call1, function_call2]),
        MagicMock(output=[message]),
    ]

    
    assistant = ChatAssistant(
        tools=mock_tools,
        developer_prompt="You are a helpful assistant.",
        chat_interface=mock_interface,
        llm_client=mock_llm_client
    )
    assistant.run()

    # Check function_call called for each function_call entry
    assert mock_tools.function_call.call_count == 2
    mock_tools.function_call.assert_has_calls([
        call(function_call1),
        call(function_call2)
    ])

    # Check display_function_call called for each function call
    assert mock_interface.display_function_call.call_count == 2
    mock_interface.display_function_call.assert_has_calls([
        call(function_call1, call1),
        call(function_call2, call2),
    ])

    # Check display_response called for the message
    assert mock_interface.display_response.call_count == 1
    mock_interface.display_response.assert_called_with(message)

    # Check input() called at least twice ("ask" and "stop")
    assert mock_interface.input.call_count >= 2


def test_chatassistant_order_message_and_function_calls():
    mock_interface = MagicMock()
    mock_interface.input.side_effect = ['ask', 'stop']

    mock_llm_client = MagicMock()

    message1 = AttrDict({'type': 'message', 'content': [{'text': 'First message.'}]})
    function_call1 = AttrDict({'type': 'function_call', 'name': 'func1', 'arguments': '{}'})
    function_call2 = AttrDict({'type': 'function_call', 'name': 'func2', 'arguments': '{}'})
    message2 = AttrDict({'type': 'message', 'content': [{'text': 'Second message.'}]})

    # The LLM first returns a message and two function calls, then a message
    mock_llm_client.send_request.side_effect = [
        MagicMock(output=[message1, function_call1, function_call2]),
        MagicMock(output=[message2]),
    ]

    mock_tools = MagicMock()

    # Simulate function_call returning a result for each call
    call1 = {"role": "function", "content": "result_func1"}
    call2 = {"role": "function", "content": "result_func2"}
    mock_tools.function_call.side_effect = [
        call1,
        call2,
    ]

    assistant = ChatAssistant(
        tools=mock_tools,
        developer_prompt="You are a helpful assistant.",
        chat_interface=mock_interface,
        llm_client=mock_llm_client
    )
    assistant.run()

    # Check input() called twice
    assert mock_interface.input.call_count == 2

    # Check LLM called twice
    assert mock_llm_client.send_request.call_count == 2

    # Check function_call called for each function_call entry
    assert mock_tools.function_call.call_count == 2
    mock_tools.function_call.assert_has_calls([
        call(function_call1),
        call(function_call2)
    ])

    # Check display_response called for each message
    assert mock_interface.display_response.call_count == 2
    mock_interface.display_response.assert_has_calls([
        call(message1),
        call(message2)]
    )

    # Check display_function_call called for each function call
    assert mock_interface.display_function_call.call_count == 2
    mock_interface.display_function_call.assert_has_calls([
        call(function_call1, call1),
        call(function_call2, call2),
    ])

    # Check the order of all relevant calls
    expected_order = [
        call.input(),
        call.display_response(message1),
        call.display_function_call(function_call1, call1),
        call.display_function_call(function_call2, call2),
        call.display_response(message2),
        call.input(),
    ]
    actual_calls = [c for c in mock_interface.mock_calls if c in expected_order]
    assert actual_calls == expected_order 