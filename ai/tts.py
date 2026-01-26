from typing import Optional

from configs.tts_config import TTSConfig
from provider.tts.base_tts import BaseTTS

from questin.schema.context import Context


def TTS(
    tts_config: Optional[TTSConfig] = None,
    context: Optional[Context] = None,
) -> BaseTTS:
    ctx = context or Context()
    if tts_config is None:
        return ctx.tts_with_cost_manager_from_tts_config(ctx.config.tts)
    return ctx.tts()
