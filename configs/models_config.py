#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models_config.py

This module defines the ModelsConfig class for handling configuration of LLM models.

Attributes:
    CONFIG_ROOT (Path): Root path for configuration files.
    METAGPT_ROOT (Path): Root path for MetaGPT files.

Classes:
    ModelsConfig (YamlModel): Configuration class for LLM models.
"""
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator

from configs.config import merge_dict
from configs.llm_config import LLMConfig
from configs.embedding_config import EmbeddingConfig
from configs.stt_config import STTConfig
from configs.tts_config import TTSConfig
from utils.constants import ROOT_PATH, CONFIG_PATH
from utils.yaml_model import YamlModel


class ModelsConfig(YamlModel):
    """
    Configuration class for `models` in `config2.yaml`.

    Attributes:
        models (Dict[str, LLMConfig]): Dictionary mapping model names or types to LLMConfig objects.

    Methods:
        update_llm_model(cls, value): Validates and updates LLM model configurations.
        from_home(cls, path): Loads configuration from ~/.metagpt/config2.yaml.
        default(cls): Loads default configuration from predefined paths.
        get(self, name_or_type: str) -> Optional[LLMConfig]: Retrieves LLMConfig by name or API type.
    """

    models: Dict[str, LLMConfig] = Field(default_factory=dict)
    embeddings: Dict[str, EmbeddingConfig] = Field(default_factory=dict)
    stts: Dict[str, STTConfig] = Field(default_factory=dict)
    tts_s: Dict[str, TTSConfig] = Field(default_factory=dict)

    @field_validator("models", mode="before")
    @classmethod
    def update_llm_model(cls, value):
        """
        Validates and updates LLM model configurations.

        Args:
            value (Dict[str, Union[LLMConfig, dict]]): Dictionary of LLM configurations.

        Returns:
            Dict[str, Union[LLMConfig, dict]]: Updated dictionary of LLM configurations.
        """
        for key, config in value.items():
            if isinstance(config, LLMConfig):
                config.model = config.model or key
            elif isinstance(config, dict):
                config["model"] = config.get("model") or key
        return value

    @field_validator("embeddings", mode="before")
    @classmethod
    def update_embedding_model(cls, value):
        for key, config in value.items():
            if isinstance(config, EmbeddingConfig):
                config.model = config.model or key
            elif isinstance(config, dict):
                config["model"] = config.get("model") or key
        return value

    @field_validator("stts", mode="before")
    @classmethod
    def update_stt_model(cls, value):
        for key, config in value.items():
            if isinstance(config, STTConfig):
                config.model = config.model or key
            elif isinstance(config, dict):
                config["model"] = config.get("model") or key
        return value

    @field_validator("tts_s", mode="before")
    @classmethod
    def update_tts_model(cls, value):
        for key, config in value.items():
            if isinstance(config, TTSConfig):
                config.model = config.model or key
            elif isinstance(config, dict):
                config["model"] = config.get("model") or key
        return value

    @classmethod
    def from_home(cls, path):
        """
        Loads configuration from ~/.metagpt/config2.yaml.

        Args:
            path (str): Relative path to configuration file.

        Returns:
            Optional[ModelsConfig]: Loaded ModelsConfig object or None if file doesn't exist.
        """
        pathname = CONFIG_PATH / path
        if not pathname.exists():
            return None
        return ModelsConfig.from_yaml_file(pathname)

    @classmethod
    def default(cls):
        """
        Loads default configuration from predefined paths.

        Returns:
            ModelsConfig: Default ModelsConfig object.
        """
        default_config_paths: List[Path] = [
            ROOT_PATH / "configs" / "config.yaml",
            CONFIG_PATH / "config.yaml",
        ]

        dicts = [ModelsConfig.read_yaml(path) for path in default_config_paths]
        final = merge_dict(dicts)
        return ModelsConfig(**final)

    def get(self, name_or_type: str) -> LLMConfig:
        """
        Retrieves LLMConfig object by name or API type.

        Args:
            name_or_type (str): Name or API type of the LLM model.

        Returns:
            Optional[LLMConfig]: LLMConfig object if found, otherwise None.
        """
        if not name_or_type:
            raise ValueError("Model name or type is required")
        model = self.models.get(name_or_type)
        if model:
            return model
        for m in self.models.values():
            if m.api_type == name_or_type:
                return m
        raise ValueError(f"Model {name_or_type} not found")

    def get_embedding_config(self, name_or_type: str) -> EmbeddingConfig:
        if not name_or_type:
            raise ValueError("Embedding model name or type is required")
        model = self.embeddings.get(name_or_type)
        if model:
            return model
        for m in self.embeddings.values():
            if m.api_type == name_or_type:
                return m
        raise ValueError(f"Embedding model {name_or_type} not found")

    def get_stt_config(self, name_or_type: str) -> STTConfig:
        if not name_or_type:
            raise ValueError("STT model name or type is required")
        model = self.stts.get(name_or_type)
        if model:
            return model
        for m in self.stts.values():
            if m.api_type == name_or_type:
                return m
        raise ValueError(f"STT model {name_or_type} not found")

    def get_tts_config(self, name_or_type: str) -> TTSConfig:
        if not name_or_type:
            raise ValueError("TTS model name or type is required")
        model = self.tts_s.get(name_or_type)
        if model:
            return model
        for m in self.tts_s.values():
            if m.api_type == name_or_type:
                return m
        raise ValueError(f"TTS model {name_or_type} not found")
