from dataclasses import dataclass
from genai_prices import calc_price, Usage, wait_prices_updated_sync
from genai_prices.data import providers
from decimal import Decimal


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
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token count cannot be negative.")

        try:
            price_data = calc_price(
                Usage(input_tokens=input_tokens, output_tokens=output_tokens), model_ref=model)
        except LookupError as le:
            raise LookupError(
                "Please check model name. Use list_all_models function to see list of supported models.")

        return CostInfo(
            input_cost=price_data.input_price, output_cost=price_data.output_price, total_cost=price_data.total_price
        )

    def list_all_models(self, provider_filter: str = None):
        """Lists all available models which has price data.

        :param str provider_filter: Name of model provider to filter by, defaults to None
        """
        providers_to_display = providers

        if provider_filter:
            provider_map = {p.id: p for p in providers}
            if provider_filter in provider_map:
                providers_to_display = [provider_map[provider_filter]]
            else:
                print(
                    f'Error: provider {provider_filter} not found in {sorted(provider_map)}')
                return None

        for provider in providers_to_display:
            print(f'{provider.name}: ({len(provider.models)} models)')
            for model in provider.models:
                model_display = f'  {provider.id}:{model.id}'
                if model.name:
                    model_display += f': {model.name}'
                print(model_display)

    def update_price(self, timeout: int = 10):
        """Updates price data.

        :param int timeout: Timeout, defaults to 10
        """
        if wait_prices_updated_sync(timeout):
            print("Price updated successfully.")
            return True
        print("Not able to update price at the moment. Please try again later.")
        return False


PricingConfig().list_all_models("x-ai")
