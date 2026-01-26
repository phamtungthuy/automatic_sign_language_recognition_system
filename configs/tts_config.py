from enum import Enum
from typing import Optional, List, Iterable, Dict
from pathlib import Path
from pydantic import field_validator

from utils.yaml_model import YamlModel
from utils.constants import ROOT_PATH, CONFIG_PATH


def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dicts into one, with the latter dict overwriting the former"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


class TTSType(Enum):
    OPENAI = "openai"


class TTSConfig(YamlModel):
    api_key: str = "sk-"
    api_type: TTSType = TTSType.OPENAI
    base_url: str = "https://api.openai.com/v1"
    api_version: Optional[str] = None

    model: Optional[str] = None  # also stands for DEPLOYMENT_NAME
    pricing_plan: Optional[str] = None  # Cost Settlement Plan Parameters.
    stream: bool = False

    # For Network
    proxy: Optional[str] = None

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
        dicts = [TTSConfig.read_yaml(path) for path in default_config_paths]
        dicts = [d["tts"] for d in dicts if "tts" in d]
        final = merge_dict(dicts)
        return TTSConfig(**final)
