from abc import ABC, abstractmethod
from typing import Optional, List, Union
from utils.cost_manager import CostManager, Costs
from utils.logs import logger

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel

from configs.embedding_config import EmbeddingConfig


class BaseEmbedding(ABC):
    config: EmbeddingConfig

    cost_manager: Optional[CostManager] = None
    aclient: Optional[AsyncOpenAI] = None
    model: Optional[str] = None
    pricing_plan: Optional[str] = None

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        raise NotImplementedError("encode method not implemented")

    @abstractmethod
    async def batch_encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        raise NotImplementedError("batch_encode method not implemented")

    def _calc_usage(self, texts: Union[str, List[str]]) -> dict:
        """Calculate token usage for embedding requests

        Args:
            texts: Input texts for embedding

        Returns:
            dict: Usage information with prompt_tokens and completion_tokens
        """
        try:
            from utils.token_counter import count_output_tokens

            if isinstance(texts, str):
                texts = [texts]

            total_tokens = 0
            model = (
                self.pricing_plan or self.model or "text-embedding-ada-002"
            )  # default model
            for text in texts:
                total_tokens += count_output_tokens(text, model)

            return {
                "prompt_tokens": total_tokens,
                "completion_tokens": 0,  # Embedding models don't have completion tokens
                "total_tokens": total_tokens,
            }
        except Exception as e:
            logger.warning(f"Usage calculation failed: {e}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _update_costs(
        self,
        usage: Union[dict, BaseModel],
        model: Optional[str] = None,
        local_calc_usage: bool = True,
    ):
        """update each request's token cost for embedding models
        Args:
            usage (dict|BaseModel): usage information containing prompt_tokens
            model (str): model name or in some scenarios called endpoint
            local_calc_usage (bool): some models don't calculate usage, it will overwrite EmbeddingConfig.calc_usage
        """
        calc_usage = getattr(self.config, "calc_usage", True) and local_calc_usage
        model_name = model or self.pricing_plan or self.model
        if not model_name:
            logger.warning("No model specified for cost calculation")
            return

        usage = usage.model_dump() if isinstance(usage, BaseModel) else usage
        if calc_usage and self.cost_manager and usage:
            try:
                # For embedding models, only prompt_tokens matter (completion_tokens is usually 0)
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                self.cost_manager.update_cost(
                    prompt_tokens, completion_tokens, model_name
                )
            except Exception as e:
                logger.error(
                    f"{self.__class__.__name__} updates costs failed! exp: {e}"
                )

    def get_costs(self) -> Costs:
        """Get cost information from cost manager

        Returns:
            Costs: NamedTuple containing token and cost information
        """
        if not self.cost_manager:
            return Costs(0, 0, 0, 0)
        return self.cost_manager.get_costs()
