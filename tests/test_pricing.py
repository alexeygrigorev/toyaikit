from toyaikit.pricing import PricingConfig
from genai_prices import calc_price, Usage
import pytest


class TestPricingConfig:

    def setup_method(self):
        self.pricing_config = PricingConfig()

    def test_calculate_cost_basic(self):
        """Test working of calculate cost function.
        """
        input_tokens = 1000
        output_tokens = 0
        model = 'gpt-5'

        genai_result = calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens), model_ref=model)

        pricing_config_result = self.pricing_config.calculate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens)

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
            self.pricing_config.calculate_cost(
                model=model, input_tokens=input_tokens, output_tokens=output_tokens)

    def test_list_all_models(self):
        """Test list all models function.
        """
        model_dict = self.pricing_config.list_all_models()
        assert isinstance(model_dict, dict)
        assert len(model_dict) > 0
        for provider, models in model_dict.items():
            assert isinstance(models, list)
            assert len(models) > 0
