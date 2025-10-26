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
            if ':' in model:
                provider, model = model.rsplit(':', maxsplit=1)

            token_usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            price_data = calc_price(
                token_usage,
                provider_id=provider,
                model_ref=model
            )

        except LookupError as le:
            raise LookupError(
                "Please check model name. Use list_all_models function to see list of supported models.", le)

        return CostInfo(
            input_cost=price_data.input_price, output_cost=price_data.output_price, total_cost=price_data.total_price
        )

    def all_available_models(self):
        """Lists all available models which has price data.

        :return dict: Dictionary with provider as key and list of models as value
        """

        model_dict = {}

        for provider in genai_data.providers:
            model_dict[provider.id] = []
            for model in provider.models:
                model_name = f'{model.id}'
                model_dict[provider.id].append(model_name)

        return model_dict
