import os
from typing import Dict, Iterable
from pydantic import Field
from utils.yaml_model import YamlModel
from configs.llm_config import LLMConfig
from configs.workspace_config import WorkspaceConfig
from configs.embedding_config import EmbeddingConfig
from utils.constants import CONFIG_PATH, ROOT_PATH
from configs.stt_config import STTConfig
from configs.tts_config import TTSConfig


class Config(YamlModel):
    llm: LLMConfig

    embedding: EmbeddingConfig
    stt: STTConfig
    tts: TTSConfig
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    repair_llm_output: bool = Field(default=False)

    @classmethod
    def default(cls, reload: bool = False, **kwargs) -> "Config":
        default_config_paths = (
            ROOT_PATH / "configs" / "config.yaml",
            CONFIG_PATH / "config.yaml",
        )
        if reload or default_config_paths not in _CONFIG_CACHE:
            dicts = [
                dict(os.environ),
                *(Config.read_yaml(path) for path in default_config_paths),
                kwargs,
            ]
            final = merge_dict(dicts)
            _CONFIG_CACHE[default_config_paths] = Config(**final)
        return _CONFIG_CACHE[default_config_paths]


def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dicts into one, with the latter dict overwriting the former"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


_CONFIG_CACHE = {}
