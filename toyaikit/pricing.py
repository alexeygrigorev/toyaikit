from dataclasses import dataclass


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class CostInfo:
    input_cost: float
    output_cost: float
    total_cost: float


class PricingConfig:
    _instance = None
    _pricing = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_pricing()
        return cls._instance

    def _load_pricing(self):
        self._pricing = PRICING

    def get_pricing(self, model: str):
        if model in self._pricing["models"]:
            return self._pricing["models"][model]
        return self._pricing["default"]

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int):
        pricing = self.get_pricing(model)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        return CostInfo(
            input_cost=input_cost, output_cost=output_cost, total_cost=total_cost
        )


PRICING = {
    "models": {
        # OpenAI Models
        "gpt-4": {
            "input": 30.0,
            "output": 60.0,
        },
        "gpt-4-turbo": {
            "input": 10.0,
            "output": 30.0,
        },
        "gpt-4o": {
            "input": 2.5,
            "output": 10.0,
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.6,
        },
        "gpt-3.5-turbo": {
            "input": 0.5,
            "output": 1.5,
        },
        "gpt-5": {
            "input": 1.25,
            "output": 10.0,
        },
        "gpt-5-mini": {
            "input": 0.25,
            "output": 2.0,
        },
        "gpt-5-nano": {
            "input": 0.05,
            "output": 0.4,
        }
    },
    "default": {
        "input": 1.0,
        "output": 3.0,
    },
}

