import os
import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
from provider.llm_provider_registry import create_llm_instance
from utils.cost_manager import CostManager, TokenCostManager, FireworksCostManager
from provider.llm.base_llm import BaseLLM
from provider.embedding.base_embedding import BaseEmbedding
from provider.embedding_provider_registry import create_embedding_instance
from configs.llm_config import LLMConfig, LLMType
from configs.embedding_config import EmbeddingConfig, EmbeddingType
from configs.config import Config
from configs.stt_config import STTConfig
from configs.tts_config import TTSConfig
from ai.settings import cost_manager, token_cost_manager, fireworks_cost_manager

from provider.stt.base_stt import BaseSTT
from provider.stt_provider_registry import create_stt_instance
from provider.tts.base_tts import BaseTTS
from provider.tts_provider_registry import create_tts_instance


class AttrDict(BaseModel):
    """A dict-like object that allows access to keys as attributes, compatible with Pydantic."""

    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        return self.__dict__.get(key, None)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def set(self, key, val: Any):
        self.__dict__[key] = val

    def get(self, key, default: Any = None):
        return self.__dict__.get(key, default)

    def remove(self, key):
        if key in self.__dict__:
            self.__delattr__(key)


class Context(BaseModel):
    """Env context"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kwargs: AttrDict = AttrDict()
    config: Config = Field(default_factory=Config.default)
    cost_manager: CostManager = cost_manager

    _llm: Optional[BaseLLM] = None

    def new_environ(self):
        """Return a new os.environ object"""
        env = os.environ.copy()
        # i = self.options
        # env.update({k: v for k, v in i.items() if isinstance(v, str)})
        return env

    def _select_costmanager(self, llm_config: LLMConfig) -> CostManager:
        """Return a CostManager instance"""
        if llm_config.api_type == LLMType.FIREWORKS:
            return fireworks_cost_manager
        elif llm_config.api_type == LLMType.OPEN_LLM:
            return token_cost_manager
        else:
            return self.cost_manager

    def embedding(self) -> BaseEmbedding:
        self._embedding = create_embedding_instance(self.config.embedding)
        if self._embedding.cost_manager is None:
            self._embedding.cost_manager = self.cost_manager
        return self._embedding

    def llm(self) -> BaseLLM:
        """Return a LLM instance, fixme: support cache"""
        # if self._llm is None:
        self._llm = create_llm_instance(self.config.llm)
        if self._llm.cost_manager is None:
            self._llm.cost_manager = self._select_costmanager(self.config.llm)
        return self._llm

    def stt(self) -> BaseSTT:
        self._stt = create_stt_instance(self.config.stt)
        if self._stt.cost_manager is None:
            self._stt.cost_manager = self.cost_manager
        return self._stt

    def tts(self) -> BaseTTS:
        self._tts = create_tts_instance(self.config.tts)
        if self._tts.cost_manager is None:
            self._tts.cost_manager = self.cost_manager
        return self._tts

    def embedding_with_cost_manager_from_embedding_config(
        self, embedding_config: EmbeddingConfig
    ) -> BaseEmbedding:
        embedding = create_embedding_instance(embedding_config)
        if embedding.cost_manager is None:
            embedding.cost_manager = self.cost_manager
        return embedding

    def llm_with_cost_manager_from_llm_config(self, llm_config: LLMConfig) -> BaseLLM:
        """Return a LLM instance, fixme: support cache"""
        # if self._llm is None:
        llm = create_llm_instance(llm_config)
        if llm.cost_manager is None:
            llm.cost_manager = self._select_costmanager(llm_config)
        return llm

    def stt_with_cost_manager_from_stt_config(self, stt_config: STTConfig) -> BaseSTT:
        stt = create_stt_instance(stt_config)
        if stt.cost_manager is None:
            stt.cost_manager = self.cost_manager
        return stt

    def tts_with_cost_manager_from_tts_config(self, tts_config: TTSConfig) -> BaseTTS:
        tts = create_tts_instance(tts_config)
        if tts.cost_manager is None:
            tts.cost_manager = self.cost_manager
        return tts

    def serialize(self) -> Dict[str, Any]:
        """Serialize the object's attributes into a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing serialized data.
        """
        return {
            "kwargs": {k: v for k, v in self.kwargs.__dict__.items()},
            "cost_manager": self.cost_manager.model_dump_json(),
        }

    def deserialize(self, serialized_data: Dict[str, Any]):
        """Deserialize the given serialized data and update the object's attributes accordingly.

        Args:
            serialized_data (Dict[str, Any]): A dictionary containing serialized data.
        """
        if not serialized_data:
            return
        kwargs = serialized_data.get("kwargs")
        if kwargs:
            for k, v in kwargs.items():
                self.kwargs.set(k, v)
        cost_manager = serialized_data.get("cost_manager")
        if cost_manager:
            self.cost_manager.model_validate_json(cost_manager)
