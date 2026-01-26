from typing import Optional

from configs.stt_config import STTConfig
from provider.stt.base_stt import BaseSTT

from questin.schema.context import Context


def STT(
    stt_config: Optional[STTConfig] = None,
    context: Optional[Context] = None,
) -> BaseSTT:
    ctx = context or Context()
    if stt_config is None:
        return ctx.stt_with_cost_manager_from_stt_config(ctx.config.stt)
    return ctx.stt()
