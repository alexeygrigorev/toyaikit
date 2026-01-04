from dataclasses import dataclass
from decimal import Decimal

from genai_prices import Usage, calc_price
from genai_prices import data as genai_data


@dataclass
class TokenUsage:
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class CostInfo:
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal

    @classmethod
    def create(cls, input_cost: Decimal, output_cost: Decimal) -> "CostInfo":
        return CostInfo(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=(input_cost + output_cost)
        )

    def plus(self, other: "CostInfo") -> "CostInfo":
        return CostInfo(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
        )

    def __add__(self, other: "CostInfo") -> "CostInfo":
        return self.plus(other)


class PricingConfig:
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int):
        """Calculate cost for a LLM API call based on token usage.

        :param str model: Name of LLM model
        :param int input_tokens: Number of input tokens
        :param int output_tokens: Number of output tokens
        :return CostInfo: Object containing input cost, ouput cost and total cost
        """
        try:
            provider = None
            if ":" in model:
                provider, model = model.rsplit(":", maxsplit=1)

            token_usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            price_data = calc_price(token_usage, provider_id=provider, model_ref=model)

            return CostInfo(
                input_cost=price_data.input_price,
                output_cost=price_data.output_price,
                total_cost=price_data.total_price,
            )

        except LookupError as le:
            # Try fallback pricing for models not in genai_prices
            model_key = model.lower()
            if model_key in FALLBACK_PRICING:
                pricing = FALLBACK_PRICING[model_key]
                # Prices are per 1M tokens, so divide by 1M
                input_cost = (pricing["input"] * input_tokens) / Decimal("1000000")
                output_cost = (pricing["output"] * output_tokens) / Decimal("1000000")
                return CostInfo.create(input_cost=input_cost, output_cost=output_cost)
            
            raise LookupError(
                "Please check model name. Use list_all_models function to see list of supported models.",
                le,
            )

    def all_available_models(self):
        """Lists all available models which has price data.

        :return dict: Dictionary with provider as key and list of models as value
        """

        model_dict = {}

        for provider in genai_data.providers:
            model_dict[provider.id] = []
            for model in provider.models:
                model_name = f"{model.id}"
                model_dict[provider.id].append(model_name)

        return model_dict


# Fallback pricing for models not in genai_prices (per 1M tokens)
FALLBACK_PRICING = {
    # Z.ai: https://docs.z.ai/guides/overview/pricing
    "glm-4.7": {"input": Decimal("0.6"), "output": Decimal("2.2")},
    "glm-4.6": {"input": Decimal("0.6"), "output": Decimal("2.2")},
    "glm-4.5": {"input": Decimal("0.6"), "output": Decimal("2.2")},
    "glm-4.5v": {"input": Decimal("0.6"), "output": Decimal("1.8")},
    "glm-4.5-x": {"input": Decimal("2.2"), "output": Decimal("8.9")},
    "glm-4.5-air": {"input": Decimal("0.2"), "output": Decimal("4.5")},
}

