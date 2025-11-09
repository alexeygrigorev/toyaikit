from decimal import Decimal

import pytest
from genai_prices import Usage, calc_price

from toyaikit.pricing import PricingConfig, CostInfo


class TestPricingConfig:
    def setup_method(self):
        self.pricing_config = PricingConfig()

    def test_calculate_cost_basic(self):
        """Test working of calculate cost function."""
        input_tokens = 1000
        output_tokens = 500
        model = "gpt-5"

        genai_result = calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            model_ref=model,
        )

        pricing_config_result = self.pricing_config.calculate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )

        assert pricing_config_result.input_cost == genai_result.input_price
        assert pricing_config_result.output_cost == genai_result.output_price
        assert pricing_config_result.total_cost == genai_result.total_price

    def test_calculate_cost_gpt_4o_mini(self):
        """Test working of calculate cost function."""
        input_tokens = 40000
        output_tokens = 1500
        model = "gpt-4o-mini"

        pricing_config_result = self.pricing_config.calculate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )

        assert pricing_config_result.input_cost == Decimal("0.006")
        assert pricing_config_result.output_cost == Decimal("0.0009")
        assert pricing_config_result.total_cost == Decimal("0.0009") + Decimal("0.006")

    def test_calculate_cost_openai_gpt_4o_mini(self):
        """Test working of calculate cost function."""
        input_tokens = 40000
        output_tokens = 1500
        model = "openai:gpt-4o-mini"

        pricing_config_result = self.pricing_config.calculate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )

        assert pricing_config_result.input_cost == Decimal("0.006")
        assert pricing_config_result.output_cost == Decimal("0.0009")
        assert pricing_config_result.total_cost == Decimal("0.0009") + Decimal("0.006")

    def test_calculate_cost_anthropic_claude_sonnet(self):
        """Test working of calculate cost function."""
        input_tokens = 40000
        output_tokens = 1500
        model = "anthropic:claude-sonnet-4-5-20250929"

        pricing_config_result = self.pricing_config.calculate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )

        assert pricing_config_result.input_cost == Decimal("0.12")
        assert pricing_config_result.output_cost == Decimal("0.0225")
        assert pricing_config_result.total_cost == Decimal("0.0225") + Decimal("0.12")

    def test_calculate_cost_wrong_model(self):
        """Test calculate cost with wrong model name."""
        input_tokens = 500
        output_tokens = 1000
        model = "IamBatman"

        with pytest.raises(LookupError):
            self.pricing_config.calculate_cost(
                model=model, input_tokens=input_tokens, output_tokens=output_tokens
            )

    def test_list_all_models(self):
        """Test list all models function."""
        model_dict = self.pricing_config.all_available_models()
        assert isinstance(model_dict, dict)
        assert len(model_dict) > 0
        for provider, models in model_dict.items():
            assert isinstance(models, list)
            assert len(models) > 0


    def test_create_cost_info(self):
        cf = CostInfo.create(
            input_cost=Decimal('0.01'),
            output_cost=Decimal('0.02')
        )
        assert cf.total_cost == Decimal('0.03')

    def test_cost_info_add(self):
        c1 = CostInfo.create(
            input_cost=Decimal('0.01'),
            output_cost=Decimal('0.10')
        )
        c2 = CostInfo.create(
            input_cost=Decimal('0.02'),
            output_cost=Decimal('0.20')
        )
        c3 = c1 + c2

        assert c3.input_cost == Decimal('0.03')
        assert c3.output_cost == Decimal('0.30')
        assert c3.total_cost == c1.total_cost + c2.total_cost
        
        