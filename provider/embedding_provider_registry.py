from configs.embedding_config import EmbeddingConfig, EmbeddingType

from provider.embedding.base_embedding import BaseEmbedding


class EmbeddingProviderRegistry:
    def __init__(self):
        self.providers = {}

    def register(self, key, provider_cls):
        self.providers[key] = provider_cls

    def get_provider(self, enum: EmbeddingType):
        """get provider instance according to the enum"""
        return self.providers[enum]


def register_embedding_provider(keys):
    """register provider to registry"""

    def decorator(cls):

        if isinstance(keys, list):
            for key in keys:
                EMBEDDING_REGISTRY.register(key, cls)
        else:
            EMBEDDING_REGISTRY.register(keys, cls)
        return cls

    return decorator


def create_embedding_instance(config: EmbeddingConfig) -> BaseEmbedding:
    """get the default embedding provider"""
    embedding = EMBEDDING_REGISTRY.get_provider(config.api_type)(config)
    return embedding


EMBEDDING_REGISTRY = EmbeddingProviderRegistry()
