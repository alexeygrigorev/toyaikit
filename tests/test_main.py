from unittest.mock import Mock, patch

from toyaikit.chat.chat import ChatAssistant
from toyaikit.chat.interface import IPythonChatInterface
from toyaikit.llm import OpenAIClient
from toyaikit.main import init
from toyaikit.tools import Tools


class TestInit:
    def test_init_with_default_parameters(self):
        """Test init() with default parameters creates all components correctly."""
        with patch("toyaikit.main.OpenAI") as mock_openai_class:
            mock_openai_instance = Mock()
            mock_openai_class.return_value = mock_openai_instance

            result = init("test prompt")

            # Verify result is ChatAssistant
            assert isinstance(result, ChatAssistant)

            # Verify OpenAI client was created with default params
            mock_openai_class.assert_called_once_with()

            # Verify runner exists and has correct type
            assert hasattr(result, "runner")

            # Verify runner components are properly configured
            assert isinstance(result.runner.tools, Tools)
            assert result.runner.developer_prompt == "test prompt"
            assert isinstance(result.runner.chat_interface, IPythonChatInterface)
            assert isinstance(result.runner.llm_client, OpenAIClient)

            # Verify LLM client configuration
            assert result.runner.llm_client.model == "gpt-4o-mini"
            assert result.runner.llm_client.client == mock_openai_instance

    def test_init_with_custom_model(self):
        """Test init() with custom model parameter."""
        with patch("toyaikit.main.OpenAI") as mock_openai_class:
            mock_openai_instance = Mock()
            mock_openai_class.return_value = mock_openai_instance

            result = init("test prompt", model="gpt-4")

            assert isinstance(result, ChatAssistant)
            assert result.runner.llm_client.model == "gpt-4"

    def test_init_with_custom_client(self):
        """Test init() with custom OpenAI client."""
        custom_client = Mock()

        result = init("test prompt", client=custom_client)

        assert isinstance(result, ChatAssistant)
        assert result.runner.llm_client.client == custom_client

    def test_init_with_all_custom_parameters(self):
        """Test init() with all parameters customized."""
        custom_client = Mock()

        result = init(
            developer_prompt="custom prompt",
            model="gpt-3.5-turbo",
            client=custom_client,
        )

        assert isinstance(result, ChatAssistant)
        assert result.runner.developer_prompt == "custom prompt"
        assert result.runner.llm_client.model == "gpt-3.5-turbo"
        assert result.runner.llm_client.client == custom_client

    def test_init_component_types(self):
        """Test that init() creates the correct component types."""
        with patch("toyaikit.main.OpenAI"):
            result = init("test prompt")

            # Verify runner exists
            assert hasattr(result, "runner")

            # Verify all expected components exist in runner and have correct types
            assert hasattr(result.runner, "tools")
            assert hasattr(result.runner, "developer_prompt")
            assert hasattr(result.runner, "chat_interface")
            assert hasattr(result.runner, "llm_client")

            assert isinstance(result.runner.tools, Tools)
            assert isinstance(result.runner.chat_interface, IPythonChatInterface)
            assert isinstance(result.runner.llm_client, OpenAIClient)

    @patch("toyaikit.main.OpenAI")
    def test_init_openai_client_not_called_when_provided(self, mock_openai_class):
        """Test that OpenAI() is not called when client is provided."""
        custom_client = Mock()

        init("test prompt", client=custom_client)

        # OpenAI class should not be instantiated
        mock_openai_class.assert_not_called()

    def test_init_creates_runner_instance(self):
        """Test that init() creates a runner with the correct type."""
        with patch("toyaikit.main.OpenAI"):
            result = init("test prompt")

            # Check that runner exists and can be called
            assert hasattr(result, "runner")
            assert hasattr(result.runner, "run")
            assert callable(result.runner.run)
