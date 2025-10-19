from toyaikit.pricing import PricingConfig
from genai_prices import calc_price, Usage
from unittest.mock import patch
from decimal import Decimal
import pytest
from io import StringIO
import sys

class TestPricingConfig:

    def setup_method(self):
        self.pricing_config = PricingConfig()

    def test_singleton_creation(self):
        """Test only one instance is created.
        """
        pricing_config2 = PricingConfig()

        assert self.pricing_config is pricing_config2
 
    def test_calculate_cost_basic(self):
        """Test working of calculate cost function.
        """
        input_tokens = 1000
        output_tokens = 0
        model = 'gpt-5'

        genai_result = calc_price(
                Usage(input_tokens=input_tokens, output_tokens=output_tokens), model_ref=model)
        
        pricing_config_result = self.pricing_config.calculate_cost(model=model, input_tokens=input_tokens, output_tokens=output_tokens)

        assert pricing_config_result.input_cost == genai_result.input_price
        assert pricing_config_result.output_cost == genai_result.output_price
        assert pricing_config_result.total_cost == genai_result.total_price

    def test_calculate_cost_wrong_model(self):
        """Test calculate cost with wrong model name.
        """
        input_tokens = 500
        output_tokens = 1000
        model = 'IamBatman'

        with pytest.raises(LookupError):
            self.pricing_config.calculate_cost(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
    
    def test_calculate_cost_negative_token(self):
        """Test calculate cost with negative token value.
        """
        input_tokens = -500
        output_tokens = -1000
        model = ''

        with pytest.raises(ValueError):
            self.pricing_config.calculate_cost(model=model, input_tokens=input_tokens, output_tokens=output_tokens)

    def test_list_all_models_with_provider_filter(self):
        """Test listing models with provider filter"""

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            self.pricing_config.list_all_models(provider_filter="openai")
            output = captured_output.getvalue()
            
            assert "openai:gpt-4o-mini: gpt 4o mini\n" in output
            assert "anthropic:claude-opus-4-0: Claude Opus 4\n" not in output
        finally:
            sys.stdout = sys.__stdout__

    def test_list_all_models_with_wrong_provider_filter(self):
        """Test listing models with wrong provider filter"""

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            self.pricing_config.list_all_models(provider_filter="IamBatman")
            output = captured_output.getvalue()
            
            assert "openai:gpt-4o-mini: gpt 4o mini\n" in output
            assert "anthropic:claude-opus-4-0: Claude Opus 4\n" in output
            assert "x-ai:grok-3: Grok 3\n" in output
        finally:
            sys.stdout = sys.__stdout__

    def test_list_all_models_with_wrong_provider_filter(self):
        """Test listing models with wrong provider filter"""

        result = self.pricing_config.list_all_models(provider_filter="IamBatman")
        assert result is None
        

    @patch('toyaikit.pricing.wait_prices_updated_sync')
    def test_update_price_success(self, mock_wait_prices):
        """Test successful price update and if default time out is 10."""
        
        mock_wait_prices.return_value = True
        result = self.pricing_config.update_price()
        
        mock_wait_prices.assert_called_once_with(10)
        assert result is True

    @patch('toyaikit.pricing.wait_prices_updated_sync')
    def test_update_price_fail(self, mock_wait_prices):
        """Test fail price update and custom timeout is getting passed."""
        
        mock_wait_prices.return_value = False
        result = self.pricing_config.update_price(15)
        
        mock_wait_prices.assert_called_with(15)
        assert result is False