from dataclasses import dataclass
from genai_prices import calc_price, Usage
from decimal import Decimal


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class CostInfo:
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal


class PricingConfig:
    _instance = None

    def __new__(cls):
        """Creates singleton class for PricingConfig.

        :return PricingConfig: Singleton instance of the PricingConfig class
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int):
        """Calculate cost for a LLM API call based on token usage.

        :param str model: Name of LLM model
        :param int input_tokens: Number of input tokens
        :param int output_tokens: Number of output tokens
        :return CostInfo: Object containing input cost, ouput cost and total cost
        """
        price_data = calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens), model_ref=model)

        return CostInfo(
            input_cost=price_data.input_price, output_cost=price_data.output_price, total_cost=price_data.total_price
        )
