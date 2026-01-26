from __future__ import annotations

import json
import re
from typing import Optional, Union

from configs.llm_config import LLMConfig, LLMType
from provider.llm.base_llm import BaseLLM
from provider.llm_provider_registry import register_provider


@register_provider(
    [
        LLMType.GROQ,
    ]
)
class GroqAPI(BaseLLM):
    pass
