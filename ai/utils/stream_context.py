"""
Stream context manager for separating different types of LLM outputs
"""

import asyncio
from contextvars import ContextVar
from typing import Optional, Literal, Dict, Any
from enum import Enum


class StreamType(str, Enum):
    """Types of stream outputs"""

    THINK = "think"
    ANSWER = "answer"
    DEFAULT = "default"


# Context variable to track current stream type
CURRENT_STREAM_TYPE: ContextVar[StreamType] = ContextVar(
    "stream_type", default=StreamType.DEFAULT
)

# Context variable to track extract tags for current stream
CURRENT_EXTRACT_TAGS: ContextVar[list[str]] = ContextVar("extract_tags", default=[])

# Context variable to track metadata for current stream
CURRENT_STREAM_METADATA: ContextVar[Dict[str, Any]] = ContextVar(
    "stream_metadata", default={}
)


def get_current_stream_type() -> StreamType:
    """Get current stream type from context"""
    return CURRENT_STREAM_TYPE.get(StreamType.DEFAULT)


def set_current_stream_type(stream_type: StreamType):
    """Set current stream type in context"""
    CURRENT_STREAM_TYPE.set(stream_type)


def set_extract_tags(extract_tags: list[str]):
    """Set extract tags in current context"""
    CURRENT_EXTRACT_TAGS.set(extract_tags)


def get_current_extract_tags() -> list[str]:
    """Get extract tags from current context"""
    return CURRENT_EXTRACT_TAGS.get([])


def get_current_stream_metadata() -> Dict[str, Any]:
    """Get current stream metadata from context"""
    return CURRENT_STREAM_METADATA.get({})


def set_current_stream_metadata(metadata: Dict[str, Any]):
    """Set current stream metadata in context"""
    CURRENT_STREAM_METADATA.set(metadata)


def get_metadata_value(key: str, default: Any = None) -> Any:
    """Get a specific metadata value by key"""
    metadata = get_current_stream_metadata()
    return metadata.get(key, default)


def update_stream_metadata(**kwargs):
    """Update metadata with new key-value pairs (merges with existing)"""
    current_metadata = get_current_stream_metadata()
    updated_metadata = {**current_metadata, **kwargs}
    set_current_stream_metadata(updated_metadata)


class StreamContext:
    """Context manager for separating stream outputs by type"""

    def __init__(
        self,
        stream_type: Literal["think", "answer"] = "answer",
        extract_tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.stream_type = StreamType(stream_type)
        self.extract_tags = extract_tags or []
        self.metadata = metadata or {}
        self._stream_token = None
        self._patterns_token = None
        self._metadata_token = None

    def __enter__(self):
        """Enter synchronous context"""
        self._stream_token = CURRENT_STREAM_TYPE.set(self.stream_type)
        self._patterns_token = CURRENT_EXTRACT_TAGS.set(self.extract_tags)
        # Merge with existing metadata if any
        existing_metadata = get_current_stream_metadata()
        merged_metadata = {**existing_metadata, **self.metadata}
        self._metadata_token = CURRENT_STREAM_METADATA.set(merged_metadata)
        return self

    def __exit__(self, *args, **kwargs):
        """Exit synchronous context"""
        if self._stream_token:
            CURRENT_STREAM_TYPE.reset(self._stream_token)
        if self._patterns_token:
            CURRENT_EXTRACT_TAGS.reset(self._patterns_token)
        if self._metadata_token:
            CURRENT_STREAM_METADATA.reset(self._metadata_token)

    async def __aenter__(self):
        """Enter async context"""
        self._stream_token = CURRENT_STREAM_TYPE.set(self.stream_type)
        self._patterns_token = CURRENT_EXTRACT_TAGS.set(self.extract_tags)
        # Merge with existing metadata if any
        existing_metadata = get_current_stream_metadata()
        merged_metadata = {**existing_metadata, **self.metadata}
        self._metadata_token = CURRENT_STREAM_METADATA.set(merged_metadata)
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        """Exit async context"""
        if self._stream_token:
            CURRENT_STREAM_TYPE.reset(self._stream_token)
        if self._patterns_token:
            CURRENT_EXTRACT_TAGS.reset(self._patterns_token)
        if self._metadata_token:
            CURRENT_STREAM_METADATA.reset(self._metadata_token)


# Convenient aliases
class ThinkContext(StreamContext):
    """Context for think/reasoning outputs

    These are typically used for internal action routing and should not be shown to users.
    """

    def __init__(self, extract_tags: list[str] = []):

        super().__init__(stream_type="think", extract_tags=extract_tags)


class AnswerContext(StreamContext):
    """Context for final answer outputs

    Usually no extract tags for answers, as they should be clean user-facing text.
    """

    def __init__(self, extract_tags: list[str] = []):
        super().__init__(stream_type="answer", extract_tags=extract_tags)
