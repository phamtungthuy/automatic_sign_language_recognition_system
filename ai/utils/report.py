import os
import asyncio
import re
from typing import List
from enum import Enum
from pydantic import BaseModel, Field, PrivateAttr
from uuid import UUID, uuid4
from contextvars import ContextVar
from typing import Any, Callable, Optional, Union, Literal
from pathlib import Path
from urllib.parse import urlparse, urlunparse, unquote
import requests
from aiohttp import ClientSession, UnixConnector

from utils.logs import (
    create_llm_stream_queue,
    get_llm_stream_queue,
    create_voice_stream_queue,
    get_voice_stream_queue,
)
from utils.constants import REPORTER_DEFAULT_URL

# from agent.base.base_agent import BaseAgent

CURRENT_ROLE: ContextVar["BaseAgent"] = ContextVar("role")  # type: ignore


class BlockType(str, Enum):
    """Enumeration for different types of blocks."""

    TERMINAL = "Terminal"
    TASK = "Task"
    BROWSER = "Browser"
    BROWSER_RT = "Browser-RT"
    EDITOR = "Editor"
    GALLERY = "Gallery"
    NOTEBOOK = "Notebook"
    DOCS = "Docs"
    THOUGHT = "Thought"
    STREAM = "Stream"


END_MARKER_NAME = "end_marker"
END_MARKER_VALUE = "\x18\x19\x1b\x18\n"


class ResourceReporter(BaseModel):
    """Base class for resource reporting."""

    block: BlockType = Field(
        description="The type of block that is reporting the resource"
    )
    uuid: UUID = Field(
        default_factory=uuid4, description="The unique identifier for the resource"
    )
    enable_llm_stream: bool = Field(
        default=False,
        description="Indicates whether to connect to an LLM stream for reporting",
    )
    enable_voice_stream: bool = Field(
        default=False,
        description="Indicates whether to connect to a voice stream for reporting",
    )
    callback_url: str = Field(
        default=REPORTER_DEFAULT_URL,
        description="The URL to which the report should be sent",
    )
    _llm_task: Optional[asyncio.Task] = PrivateAttr(None)
    _voice_task: Optional[asyncio.Task] = PrivateAttr(None)

    def report(self, value: Any, name: str, extra: Optional[dict] = None):
        """Synchronously report resource observation data.

        Args:
            value: The data to report.
            name: The type name of the data.
        """
        return self._report(value, name, extra)

    async def async_report(self, value: Any, name: str, extra: Optional[dict] = None):
        """Asynchronously report resource observation data.

        Args:
            value: The data to report.
            name: The type name of the data.
        """
        return await self._async_report(value, name, extra)

    @classmethod
    def set_report_fn(cls, fn: Callable):
        """Set the synchronous report function.

        Args:
            fn: A callable function used for synchronous reporting. For example:

                >>> def _report(self, value: Any, name: str):
                ...     print(value, name)

        """
        cls._report = fn

    @classmethod
    def set_async_report_fn(cls, fn: Callable):
        """Set the asynchronous report function.

        Args:
            fn: A callable function used for asynchronous reporting. For example:

                ```python
                >>> async def _report(self, value: Any, name: str):
                ...     print(value, name)
                ```
        """
        cls._async_report = fn

    def _report(self, value: Any, name: str, extra: Optional[dict] = None):
        if not self.callback_url:
            return

        data = self._format_data(value, name, extra)
        resp = requests.post(self.callback_url, json=data)
        resp.raise_for_status()
        return resp.text

    async def _async_report(self, value: Any, name: str, extra: Optional[dict] = None):
        if not self.callback_url:
            return

        data = self._format_data(value, name, extra)
        url = self.callback_url
        _result = urlparse(url)
        sessiion_kwargs = {}
        if _result.scheme.endswith("+unix"):
            parsed_list = list(_result)
            parsed_list[0] = parsed_list[0][:-5]
            parsed_list[1] = "fake.org"
            url = urlunparse(parsed_list)
            sessiion_kwargs["connector"] = UnixConnector(path=unquote(_result.netloc))

        async with ClientSession(**sessiion_kwargs) as client:
            async with client.post(url, json=data) as resp:
                resp.raise_for_status()
                return await resp.text()

    def _format_data(self, value, name, extra):
        data = self.model_dump(mode="json", exclude={"callback_url", "llm_stream"})
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        elif isinstance(value, Path):
            value = str(value)

        if name == "path" and isinstance(value, str):
            value = os.path.abspath(value)
        data["value"] = value
        data["name"] = name
        role = CURRENT_ROLE.get(None)
        if role:
            role_name = role.name
        else:
            role_name = os.environ.get("METAGPT_ROLE")
        data["role"] = role_name
        if extra:
            data["extra"] = extra
        return data

    def __enter__(self):
        """Enter the synchronous streaming callback context."""
        return self

    def __exit__(self, *args, **kwargs):
        """Exit the synchronous streaming callback context."""
        self.report(None, END_MARKER_NAME)

    async def __aenter__(self):
        """Enter the asynchronous streaming callback context."""
        if self.enable_llm_stream:
            queue = create_llm_stream_queue()
            self._llm_task = asyncio.create_task(self._llm_stream_report(queue))
        if self.enable_voice_stream:
            queue = create_voice_stream_queue()
            self._voice_task = asyncio.create_task(self._voice_stream_report(queue))
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        """Exit the asynchronous streaming callback context."""
        if self.enable_llm_stream and exc_type != asyncio.CancelledError:
            queue = get_llm_stream_queue()
            if queue:
                await queue.put(None)
            if self._llm_task:
                await self._llm_task
            self._llm_task = None
        if self.enable_voice_stream and exc_type != asyncio.CancelledError:
            queue = get_voice_stream_queue()
            if queue:
                await queue.put(None)
            if self._voice_task:
                await self._voice_task
            self._voice_task = None
        await self.async_report(None, END_MARKER_NAME)

    async def _llm_stream_report(self, queue: asyncio.Queue):
        while True:
            data = await queue.get()
            if data is None:
                return

            if isinstance(data, dict) and "content" in data:
                content = data["content"]
                stream_type = data.get("stream_type")
                extract_tags = data.get("extract_tags", [])

                # Set context temporarily for this callback
                if stream_type or extract_tags:
                    from questin.utils.stream_context import (
                        CURRENT_STREAM_TYPE,
                        CURRENT_EXTRACT_TAGS,
                    )

                    stream_token = None
                    patterns_token = None
                    try:
                        if stream_type:
                            stream_token = CURRENT_STREAM_TYPE.set(stream_type)
                        if extract_tags:
                            patterns_token = CURRENT_EXTRACT_TAGS.set(extract_tags)
                        await self.async_report(content, "content")
                    finally:
                        if stream_token:
                            CURRENT_STREAM_TYPE.reset(stream_token)
                        if patterns_token:
                            CURRENT_EXTRACT_TAGS.reset(patterns_token)
                else:
                    await self.async_report(content, "content")
            else:
                # Backward compatibility: plain string
                await self.async_report(data, "content")

    async def _voice_stream_report(self, queue: asyncio.Queue):
        raise NotImplementedError("Voice stream reporting is not implemented")

    async def wait_voice_stream_report(self):
        """Wait for the voice stream report to complete."""
        queue = get_voice_stream_queue()
        while self._voice_task and queue:
            if queue.empty():
                break
            await asyncio.sleep(0.01)

    async def wait_llm_stream_report(self):
        """Wait for the LLM stream report to complete."""
        queue = get_llm_stream_queue()
        while self._llm_task and queue:
            if queue.empty():
                break
            await asyncio.sleep(0.01)


class TerminalReporter(ResourceReporter):
    """Terminal output callback for streaming reporting of command and output.

    The terminal has state, and an agent can open multiple terminals and input different commands into them.
    To correctly display these states, each terminal should have its own unique ID, so in practice, each terminal
    should instantiate its own TerminalReporter object.
    """

    block: Literal[BlockType.TERMINAL] = BlockType.TERMINAL

    def report(self, value: str, name: Literal["cmd", "output"]):
        """Report terminal command or output synchronously."""
        return super().report(value, name)

    async def async_report(self, value: str, name: Literal["cmd", "output"]):
        """Report terminal command or output asynchronously."""
        return await super().async_report(value, name)


class FileReporter(ResourceReporter):
    """File resource callback for reporting complete file paths.

    There are two scenarios: if the file needs to be output in its entirety at once, use non-streaming callback;
    if the file can be partially output for display first, use streaming callback.
    """

    def report(
        self,
        value: Union[Path, dict, Any],
        name: Literal["path", "meta", "content"] = "path",
        extra: Optional[dict] = None,
    ):
        """Report file resource synchronously."""
        return super().report(value, name, extra)

    async def async_report(
        self,
        value: Union[Path, dict, Any],
        name: Literal["path", "meta", "content"] = "path",
        extra: Optional[dict] = None,
    ):
        """Report file resource asynchronously."""
        return await super().async_report(value, name, extra)


class NotebookReporter(FileReporter):
    """Equivalent to FileReporter(block=BlockType.NOTEBOOK)."""

    block: Literal[BlockType.NOTEBOOK] = BlockType.NOTEBOOK


class ObjectReporter(ResourceReporter):
    """Callback for reporting complete object resources."""

    def report(self, value: dict, name: Literal["object"] = "object"):
        """Report object resource synchronously."""
        return super().report(value, name)

    async def async_report(self, value: dict, name: Literal["object"] = "object"):
        """Report object resource asynchronously."""
        return await super().async_report(value, name)


class ThoughtReporter(ObjectReporter):
    """Reporter for object resources to Task Block."""

    block: Literal[BlockType.THOUGHT] = BlockType.THOUGHT


TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9_:-]+)>")


class StreamReporter(ObjectReporter):
    block: Literal[BlockType.STREAM] = BlockType.STREAM
    callback_fn: Optional[Callable] = None
    think_callback_fn: Optional[Callable] = None
    answer_callback_fn: Optional[Callable] = None
    voice_callback_fn: Optional[Callable] = None

    _buffer: str = PrivateAttr(default="")
    _max_buffer_size: int = PrivateAttr(
        default=200
    )  # Maximum buffer size to prevent memory issues
    tag_stack: List[str] = []

    def process_xml_stream(
        self,
        extract_tags: List[str],
        value: str,
        allow_outside_text: bool = True,
    ) -> str:
        """Nhận từng token, tách text cần lấy và cập nhật buffer an toàn.

        Khác với phiên bản cũ (return sớm, không cắt buffer), hàm này:
        - Tích lũy token vào buffer.
        - Duyệt nhiều tag/text trong cùng một lần gọi.
        - Cắt bỏ phần đã tiêu thụ khỏi buffer để lần sau không lặp vô hạn.
        - Chỉ trả về chuỗi đã được phép xuất, hoặc "" nếu chưa đủ dữ liệu.
        """
        self._buffer += value
        pos = 0
        outputs: List[str] = []
        extract_set = set(extract_tags)

        def current_tag_name() -> str | None:
            return self.tag_stack[-1] if self.tag_stack else None

        while True:
            m = TAG_RE.search(self._buffer, pos)
            if not m:
                # Không còn tag hoàn chỉnh trong buffer
                remaining = self._buffer[pos:]
                if remaining:
                    # Xử lý trường hợp "<rea" ở cuối buffer
                    last_lt = remaining.rfind("<")
                    if last_lt == -1:
                        text_part = remaining
                        tail = ""
                    else:
                        after_lt = remaining[last_lt:]
                        if ">" in after_lt:
                            text_part = remaining
                            tail = ""
                        else:
                            text_part = remaining[:last_lt]
                            tail = remaining[last_lt:]

                    if text_part:
                        cur = current_tag_name()
                        if cur in extract_set or (cur is None and allow_outside_text):
                            outputs.append(text_part)
                    # Giữ lại phần chưa đủ tag cho lần sau
                    self._buffer = tail
                else:
                    self._buffer = ""
                break

            start, end = m.span()
            before = self._buffer[pos:start]
            if before:
                cur = current_tag_name()
                if cur in extract_set or (cur is None and allow_outside_text):
                    outputs.append(before)

            # Cập nhật stack theo tag gặp được
            closing, tagname = m.groups()
            if closing:
                if self.tag_stack and self.tag_stack[-1] == tagname:
                    self.tag_stack.pop()
            else:
                self.tag_stack.append(tagname)

            pos = end

        if outputs:
            return "".join(outputs)
        # Không có text hợp lệ để xuất lần này
        return ""

    async def _async_report(self, value: Any, name: str, extra: Optional[dict] = None):
        # Get current stream type and forbidden patterns from context
        from questin.utils.stream_context import (
            get_current_stream_type,
            get_current_extract_tags,
            StreamType,
        )

        stream_type = get_current_stream_type()
        extract_tags = get_current_extract_tags()

        filtered_value = value
        if isinstance(value, str) and value:
            filtered_value = self.process_xml_stream(extract_tags, value)
        elif name == END_MARKER_NAME:
            # Flush remaining buffer at the end
            if self._buffer:
                filtered_value = self._buffer
                self._buffer = ""
            else:
                filtered_value = value

        # Route to appropriate callback based on stream type
        if stream_type == StreamType.THINK and self.think_callback_fn:
            if asyncio.iscoroutinefunction(self.think_callback_fn):
                await self.think_callback_fn(filtered_value)
            else:
                self.think_callback_fn(filtered_value)
        elif stream_type == StreamType.ANSWER and self.answer_callback_fn:
            if asyncio.iscoroutinefunction(self.answer_callback_fn):
                await self.answer_callback_fn(filtered_value)
            else:
                self.answer_callback_fn(filtered_value)
        elif self.callback_fn:
            # Default callback for unspecified stream types
            if asyncio.iscoroutinefunction(self.callback_fn):
                await self.callback_fn(filtered_value)
            else:
                self.callback_fn(filtered_value)

        return await super()._async_report(filtered_value, name, extra)

    async def _async_voice_report(self, content: bytes, name: str):
        if not self.voice_callback_fn:
            return
        if asyncio.iscoroutinefunction(self.voice_callback_fn):
            await self.voice_callback_fn(content)
        else:
            self.voice_callback_fn(content)

    async def _voice_stream_report(self, queue: asyncio.Queue):
        while True:
            data = await queue.get()
            if data is None:
                return
            if isinstance(data, dict) and "content" in data:
                content = data["content"]
                await self._async_voice_report(content, "content")
