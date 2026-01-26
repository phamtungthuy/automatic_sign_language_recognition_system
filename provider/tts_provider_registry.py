from configs.tts_config import TTSConfig, TTSType
from provider.tts.base_tts import BaseTTS


class TTSProviderRegistry:
    def __init__(self):
        self.providers = {}

    def register(self, key, provider_cls):
        self.providers[key] = provider_cls

    def get_provider(self, enum: TTSType):
        """get provider instance according to the enum"""
        return self.providers[enum]


def register_tts_provider(keys):
    """register provider to registry"""

    def decorator(cls):
        if isinstance(keys, list):
            for key in keys:
                TTS_REGISTRY.register(key, cls)
        else:
            TTS_REGISTRY.register(keys, cls)
        return cls

    return decorator


def create_tts_instance(config: TTSConfig) -> BaseTTS:
    """get the default tts provider"""
    tts = TTS_REGISTRY.get_provider(config.api_type)(config)
    return tts


TTS_REGISTRY = TTSProviderRegistry()
