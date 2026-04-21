import warnings
from decimal import Decimal

import pytest
from genai_prices import Usage, calc_price

from toyaikit.pricing import PricingConfig, CostInfo, UnknownModelWarning


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
        """Test calculate cost with wrong model name returns None and warns."""
        input_tokens = 500
        output_tokens = 1000
        model = "IamBatman"

        with pytest.warns(UnknownModelWarning, match="IamBatman"):
            result = self.pricing_config.calculate_cost(
                model=model, input_tokens=input_tokens, output_tokens=output_tokens
            )

        assert result is None

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

    def test_calculate_cost_glm_fallback(self):
        """Test fallback pricing for GLM models."""
        input_tokens = 1000000  # 1M tokens
        output_tokens = 500000  # 0.5M tokens
        model = "glm-4.5"

        pricing_config_result = self.pricing_config.calculate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )

        # GLM-4.5: $0.6 per 1M input tokens, $2.2 per 1M output tokens
        assert pricing_config_result.input_cost == Decimal("0.6")
        assert pricing_config_result.output_cost == Decimal("1.1")  # 0.5M * $2.2
        assert pricing_config_result.total_cost == Decimal("1.7")

    def test_calculate_cost_glm_air_fallback(self):
        """Test fallback pricing for GLM-4.5-Air model."""
        input_tokens = 2000000  # 2M tokens
        output_tokens = 1000000  # 1M tokens
        model = "glm-4.5-air"

        pricing_config_result = self.pricing_config.calculate_cost(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )

        # GLM-4.5-Air: $0.2 per 1M input tokens, $4.5 per 1M output tokens
        assert pricing_config_result.input_cost == Decimal("0.4")  # 2M * $0.2
        assert pricing_config_result.output_cost == Decimal("4.5")  # 1M * $4.5
        assert pricing_config_result.total_cost == Decimal("4.9")

    def test_calculate_cost_groq_unknown_model(self):
        """Test that unknown Groq model returns None and warns."""
        input_tokens = 1000
        output_tokens = 500
        model = "openai/gpt-oss-20b"

        with pytest.warns(UnknownModelWarning):
            result = self.pricing_config.calculate_cost(
                model=model, input_tokens=input_tokens, output_tokens=output_tokens
            )

        assert result is None

    def test_register_model_adds_fallback_pricing(self):
        """Users can register pricing for unknown models."""
        self.pricing_config.register_model("my-custom-model", input_price=1.0, output_price=3.0)

        result = self.pricing_config.calculate_cost(
            model="my-custom-model", input_tokens=1_000_000, output_tokens=500_000
        )

        assert result.input_cost == Decimal("1.0")
        assert result.output_cost == Decimal("1.5")
        assert result.total_cost == Decimal("2.5")

    def test_register_model_does_not_warn(self):
        """Registered models should not trigger UnknownModelWarning."""
        self.pricing_config.register_model("quiet-model", input_price="0.5", output_price="1.5")

        with warnings.catch_warnings():
            warnings.simplefilter("error", UnknownModelWarning)
            result = self.pricing_config.calculate_cost(
                model="quiet-model", input_tokens=1000, output_tokens=1000
            )

        assert result is not None

    def test_register_model_is_instance_scoped(self):
        """register_model on one instance should not leak to another."""
        other = PricingConfig()
        self.pricing_config.register_model("scoped-model", input_price=1, output_price=2)

        with pytest.warns(UnknownModelWarning):
            assert other.calculate_cost("scoped-model", 100, 100) is None
