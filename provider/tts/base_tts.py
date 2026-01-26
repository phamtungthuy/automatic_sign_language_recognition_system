from abc import ABC, abstractmethod
from typing import Optional, Any, Union
from pydantic import BaseModel
from openai import AsyncOpenAI

from configs.tts_config import TTSConfig
from utils.cost_manager import CostManager
from utils.logs import logger


class BaseTTS(ABC):
    config: TTSConfig

    aclient: Optional[AsyncOpenAI] = None
    cost_manager: Optional[CostManager] = None
    model: Optional[str] = None
    pricing_plan: Optional[str] = None

    @abstractmethod
    def __init__(self, config: TTSConfig):
        raise NotImplementedError("Subclasses must implement this method")

    def _update_costs(
        self,
        usage: Union[dict, BaseModel],
        model: Optional[str] = None,
        local_calc_usage: bool = True,
    ):
        """update each request's token cost
        Args:
            model (str): model name or in some scenarios called endpoint
            local_calc_usage (bool): some models don't calculate usage, it will overwrite LLMConfig.calc_usage
        """
        calc_usage = self.config.calc_usage and local_calc_usage
        model = model or self.pricing_plan
        model = model or self.model
        usage = usage.model_dump() if isinstance(usage, BaseModel) else usage
        if calc_usage and self.cost_manager and usage:
            try:
                prompt_tokens = int(usage.get("input_tokens", 0))
                completion_tokens = int(usage.get("output_tokens", 0))
                self.cost_manager.update_cost(prompt_tokens, completion_tokens, model)
            except Exception as e:
                logger.error(
                    f"{self.__class__.__name__} updates costs failed! exp: {e}"
                )

    @abstractmethod
    async def _aaudio_speech(self, text: str, voice: str, **kwargs) -> bytes:
        pass

    @abstractmethod
    async def _aaudio_speech_stream(self, text: str, voice: str, **kwargs) -> bytes:
        pass

    @abstractmethod
    def get_audio_content(self, rsp: Any) -> bytes:
        pass

    async def aspeech(
        self, text: str, voice: str, stream: Optional[bool] = None, **kwargs
    ) -> bytes:
        if stream is None:
            stream = self.config.stream
        if stream:
            return await self._aaudio_speech_stream(text, voice, **kwargs)
        rsp = await self._aaudio_speech(text, voice, **kwargs)
        return self.get_audio_content(rsp)

    async def aspeak(
        self, text: str, voice: str, stream: Optional[bool] = None, **kwargs
    ):
        if stream is None:
            stream = self.config.stream
        rsp = await self.aspeech(text, voice, stream, **kwargs)
        return rsp
