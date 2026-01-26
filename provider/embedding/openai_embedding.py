from typing import List, Union
import numpy as np
from tqdm import tqdm

# import torch

from openai import AsyncOpenAI
from openai._base_client import AsyncHttpxClientWrapper

from configs.embedding_config import EmbeddingConfig, EmbeddingType
from provider.embedding_provider_registry import register_embedding_provider
from provider.embedding.base_embedding import BaseEmbedding


@register_embedding_provider(
    [
        EmbeddingType.OPENAI,
    ]
)
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._init_client()

    def _init_client(self):
        if not self.config.api_key or not self.config.base_url:
            raise ValueError("OpenAI client not initialized")
        self.model = self.config.model
        self.pricing_plan = self.config.pricing_plan or self.model
        kwargs = self._make_client_kwargs()
        self.aclient = AsyncOpenAI(**kwargs)

    def _make_client_kwargs(self) -> dict:
        kwargs = {"api_key": self.config.api_key, "base_url": self.config.base_url}

        # to use proxy, openai v1 needs http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)  # type: ignore

        return kwargs

    def _get_proxy_params(self) -> dict:
        params = {}
        if self.config.proxy:
            params = {"proxy": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if not self.aclient or not self.model:
            raise ValueError("OpenAI client not initialized")
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != "" else " " for t in texts]
        response = await self.aclient.embeddings.create(input=texts, model=self.model)

        # Update costs
        usage = getattr(response, "usage", None)
        if usage:
            # Convert usage to dict if it's a pydantic model
            usage_dict = (
                usage.model_dump() if hasattr(usage, "model_dump") else usage.__dict__
            )
            self._update_costs(usage_dict)
        else:
            # Calculate usage manually if not provided
            usage_dict = self._calc_usage(texts)
            self._update_costs(usage_dict)

        results = np.array([r.embedding for r in response.data])
        return results

    async def batch_encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        if not self.aclient or not self.model:
            raise ValueError("OpenAI client not initialized")
        if isinstance(texts, str):
            texts = [texts]

        batch_size = self.config.batch_size

        if len(texts) <= batch_size:
            results = await self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                try:
                    batch_result = await self.encode(batch)
                    results.append(batch_result)
                except:
                    import traceback

                    traceback.print_exc()
                pbar.update(len(batch))
            pbar.close()
            # Concatenate all batch results instead of creating array of arrays
            results = np.concatenate(results, axis=0)
        # if isinstance(results, torch.Tensor):
        #     results = results.cpu()
        #     results = results.numpy()
        if self.config.norm:
            results = (results / np.linalg.norm(results, axis=1, keepdims=True)).T

        return results
