from typing import Optional
from openai import AsyncOpenAI
from openai._legacy_response import HttpxBinaryResponseContent
from utils.cost_manager import CostManager

from configs.tts_config import TTSConfig, TTSType
from provider.tts.base_tts import BaseTTS

from provider.tts_provider_registry import register_tts_provider
from utils.logs import log_tts_stream


@register_tts_provider(
    [
        TTSType.OPENAI,
    ]
)
class OpenAITTS(BaseTTS):
    aclient: AsyncOpenAI

    def __init__(self, config: TTSConfig):
        self.config = config
        self._init_client()
        self.cost_manager: Optional[CostManager] = None

    def _init_client(self):
        self.model = self.config.model
        self.pricing_plan = self.config.pricing_plan or self.model
        kwargs = self._make_client_kwargs()
        self.aclient = AsyncOpenAI(**kwargs)

    def _get_proxy_params(self) -> dict:
        params = {}
        if self.config.proxy:
            params = {"proxy": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    def _make_client_kwargs(self) -> dict:
        kwargs = {"api_key": self.config.api_key, "base_url": self.config.base_url}
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)  # type: ignore
        return kwargs

    def _cons_kwargs(self, text: str, voice: str, **extra_kwargs) -> dict:
        kwargs = {
            "model": self.model,
            "input": text,
            "voice": voice,
            # Mặc định mp3, frontend có thể override bằng extra_kwargs
            # Options: mp3, opus, aac, flac, wav, pcm
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    def get_audio_content(self, rsp: HttpxBinaryResponseContent) -> bytes:
        return rsp.content

    def _calc_usage(self, text: str, audio_bytes: bytes) -> dict:
        input_tokens = len(text)  # hoặc count tokens thực sự

        # Estimate audio duration từ MP3 bytes
        # MP3 128kbps ~ 16KB/giây, 24kHz mono thường ~ 6KB/giây
        audio_duration_seconds = len(audio_bytes) / 6000  # rough estimate
        output_tokens = int(audio_duration_seconds * 50)  # ~50 tokens/giây (estimate)

        return {"input_tokens": input_tokens, "output_tokens": output_tokens}

    async def _aaudio_speech(
        self, text: str, voice: str, **kwargs
    ) -> HttpxBinaryResponseContent:
        rsp = await self.aclient.audio.speech.create(
            **self._cons_kwargs(text, voice, **kwargs)
        )
        self._update_costs(self._calc_usage(text, rsp.content))
        return rsp

    async def _aaudio_speech_stream(self, text: str, voice: str, **kwargs) -> bytes:

        collected_audio = bytes()
        async with self.aclient.audio.speech.with_streaming_response.create(
            **self._cons_kwargs(text, voice, **kwargs)
        ) as response:
            async for chunk in response.iter_bytes():
                collected_audio += chunk
                log_tts_stream(chunk)
        self._update_costs(self._calc_usage(text, collected_audio))
        return collected_audio
