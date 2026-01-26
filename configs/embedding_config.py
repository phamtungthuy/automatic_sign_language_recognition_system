from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pydantic import field_validator

from utils.yaml_model import YamlModel
from utils.constants import CONFIG_PATH, ROOT_PATH


def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dicts into one, with the latter dict overwriting the former"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


class EmbeddingType(Enum):
    OPENAI = "openai"


class EmbeddingConfig(YamlModel):
    """Config for Embedding.

    Examples:
    ---------
    api_type: "openai"
    api_key: "YOU_API_KEY"
    dimensions: "YOUR_MODEL_DIMENSIONS"

    api_type: "azure"
    api_key: "YOU_API_KEY"
    base_url: "YOU_BASE_URL"
    api_version: "YOU_API_VERSION"
    dimensions: "YOUR_MODEL_DIMENSIONS"

    api_type: "gemini"
    api_key: "YOU_API_KEY"

    api_type: "ollama"
    base_url: "YOU_BASE_URL"
    model: "YOU_MODEL"
    dimensions: "YOUR_MODEL_DIMENSIONS"
    """

    api_type: EmbeddingType = EmbeddingType.OPENAI
    api_key: str = "sk-"
    base_url: str = "https://api.openai.com/v1"
    api_version: Optional[str] = None

    model: str = "text-embedding-3-small"
    pricing_plan: Optional[str] = None
    batch_size: int = 16
    dimensions: Optional[int] = None  # output dimension of embedding model
    norm: bool = False  # normalize the embedding
    # For Network
    proxy: Optional[str] = None

    # For Chat Completion
    max_token: int = 8192

    # Cost Control
    calc_usage: bool = True

    @field_validator("api_type", mode="before")
    @classmethod
    def check_api_type(cls, v):
        if v == "":
            return None
        return v

    @classmethod
    def default(cls):
        default_config_paths: List[Path] = [
            ROOT_PATH / "configs/config.yaml",
            CONFIG_PATH / "config.yaml",
        ]
        dicts = [EmbeddingConfig.read_yaml(path) for path in default_config_paths]
        dicts = [d["embedding"] for d in dicts if "embedding" in d]
        final = merge_dict(dicts)
        return EmbeddingConfig(**final)
