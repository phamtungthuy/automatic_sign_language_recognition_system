import os
import json
import ast
import base64
import inspect
import contextlib
import re
import requests
import importlib
import datetime
from io import BytesIO
import uuid
from functools import partial
import aiofiles
import chardet
from bs4 import BeautifulSoup
from pathlib import Path
from pydantic_core import to_jsonable_python
from typing import Any, Optional, Callable, Union, List, Tuple, Dict
from tenacity import RetryCallState, _utils

from diffusers.utils.loading_utils import load_image
from utils.logs import logger
from utils.exceptions import handle_exception
from PIL import Image
import loguru
from utils.constants import MARKDOWN_TITLE_PREFIX, MESSAGE_ROUTE_TO_ALL


def singleton(cls):
    """Decorator to create singleton class"""
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def get_uuid():
    return uuid.uuid1().hex


def get_uuid_int() -> int:
    return abs(uuid.uuid4().int) % (2**63 - 1)


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name for Milvus.
    Milvus requires collection names to start with a letter or underscore.
    If name starts with a digit, prepend an underscore.
    """
    if name and name[0].isdigit():
        return f"_{name}"
    return name


def desanitize_collection_name(name: str) -> str:
    """
    Reverse the sanitization to get the original name.
    If name starts with underscore followed by a digit, remove the leading underscore.
    """
    if name.startswith("_") and len(name) > 1 and name[1].isdigit():
        return name[1:]
    return name


def datetime_format(date_time: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(
        date_time.year,
        date_time.month,
        date_time.day,
        date_time.hour,
        date_time.minute,
        date_time.second,
    )


def get_current_datetime() -> datetime.datetime:
    return datetime_format(datetime.datetime.now())


def image_to_bytes(img_url: str) -> bytes:
    img_byte = BytesIO()
    type = img_url.split(".")[-1]
    load_image(img_url).save(img_byte, format="png")
    img_data = img_byte.getvalue()
    return img_data


def audio_to_bytes(audio_url: str) -> bytes:
    with open(audio_url, "rb") as f:
        audio_data = f.read()
    return audio_data


def video_to_bytes(video_url: str) -> bytes:
    with open(video_url, "rb") as f:
        video_data = f.read()
    return video_data


def import_class(class_name: str, module_name: str) -> type:
    module = importlib.import_module(module_name)
    a_class = getattr(module, class_name)
    return a_class


def get_class_name(cls) -> str:
    """Return class name"""
    return f"{cls.__module__}.{cls.__name__}"


def any_to_str(val: Any) -> str:
    """Return the class name or the class name of the object, or 'val' if it's a string type."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)


def any_to_str_set(val) -> set:
    """Convert any type to string set."""
    res = set()

    # Check if the value is iterable, but not a string (since strings are technically iterable)
    if isinstance(val, (dict, list, set, tuple)):
        # Special handling for dictionaries to iterate over values
        if isinstance(val, dict):
            val = val.values()

        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))

    return res


def parse_json_code_block(markdown_text: str) -> List[str]:
    json_blocks = (
        re.findall(r"```json(.*?)```", markdown_text, re.DOTALL)
        if "```json" in markdown_text
        else [markdown_text]
    )

    return [v.strip() for v in json_blocks]


def remove_comments(code_str: str) -> str:
    """Remove comments from code."""
    pattern = r"(\".*?\"|\'.*?\')|(\#.*?$)"

    def replace_func(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    clean_code = re.sub(pattern, replace_func, code_str, flags=re.MULTILINE)
    clean_code = os.linesep.join(
        [s.rstrip() for s in clean_code.splitlines() if s.strip()]
    )
    return clean_code


def is_send_to(message: "Message", addresses: set):  # type: ignore
    """Return whether it's consumer"""
    if MESSAGE_ROUTE_TO_ALL in message.send_to:
        return True
    for i in addresses:
        if i in message.send_to:
            return True
    return False


def any_to_name(val):
    """
    Convert a value to its name by extracting the last part of the dotted path.
    """
    return any_to_str(val).split(".")[-1]


def read_json_file(json_file: str, encoding: str = "utf-8") -> Any:
    if not Path(json_file).exists():
        raise FileNotFoundError(f"json_file: {json_file} not exist, return []")

    with open(json_file, "r", encoding=encoding) as fin:
        try:
            data = json.load(fin)
        except Exception:
            raise ValueError(f"read json file: {json_file} failed")
    return data


def fix_bgroken_generated_json(json_str: str) -> str:
    """
    Fixes a malformed JSON string by:
    - Removing the last comma and any trailing content.
    - Iterating over the JSON string once to determine and fix unclosed braces or brackets.
    - Ensuring braces and brackets inside string literals are not considered.

    If the original json_str string can be successfully loaded by json.loads(), will directly return it without any modification.

    Args:
        json_str (str): The malformed JSON string to be fixed.

    Returns:
        str: The corrected JSON string.
    """

    def find_unclosed(json_str):
        """
        Identifies the unclosed braces and brackets in the JSON string.

        Args:
            json_str (str): The JSON string to analyze.

        Returns:
            list: A list of unclosed elements in the order they were opened.
        """
        unclosed = []
        inside_string = False
        escape_next = False

        for char in json_str:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    inside_string = False
            else:
                if char == '"':
                    inside_string = True
                elif char in "{[":
                    unclosed.append(char)
                elif char in "}]":
                    if unclosed and (
                        (char == "}" and unclosed[-1] == "{")
                        or (char == "]" and unclosed[-1] == "[")
                    ):
                        unclosed.pop()

        return unclosed

    try:
        # Try to load the JSON to see if it is valid
        json.loads(json_str)
        return json_str  # Return as-is if valid
    except json.JSONDecodeError as e:
        pass

    # Step 1: Remove trailing content after the last comma.
    last_comma_index = json_str.rfind(",")
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]

    # Step 2: Identify unclosed braces and brackets.
    unclosed_elements = find_unclosed(json_str)

    # Step 3: Append the necessary closing elements in reverse order of opening.
    closing_map = {"{": "}", "[": "]"}
    for open_char in reversed(unclosed_elements):
        json_str += closing_map[open_char]

    return json_str


def handle_unknown_serialization(x: Any) -> str:
    """For `to_jsonable_python` debug, get more detail about the x."""

    if inspect.ismethod(x):
        tip = f"Cannot serialize method '{x.__func__.__name__}' of class '{x.__self__.__class__.__name__}'"
    elif inspect.isfunction(x):
        tip = f"Cannot serialize function '{x.__name__}'"
    elif hasattr(x, "__class__"):
        tip = f"Cannot serialize instance of '{x.__class__.__name__}'"
    elif hasattr(x, "__name__"):
        tip = f"Cannot serialize class or module '{x.__name__}'"
    else:
        tip = f"Cannot serialize object of type '{type(x).__name__}'"

    raise TypeError(tip)


def write_json_file(
    json_file: str,
    data: Any,
    encoding: str = "utf-8",
    indent: int = 4,
    use_fallback: bool = False,
):
    folder_path = Path(json_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    custom_default = partial(
        to_jsonable_python,
        fallback=handle_unknown_serialization if use_fallback else None,
    )

    with open(json_file, "w", encoding=encoding) as fout:
        json.dump(data, fout, ensure_ascii=False, indent=indent, default=custom_default)


def read_jsonl_file(jsonl_file: str, encoding="utf-8") -> list[dict]:
    if not Path(jsonl_file).exists():
        raise FileNotFoundError(f"json_file: {jsonl_file} not exist, return []")
    datas = []
    with open(jsonl_file, "r", encoding=encoding) as fin:
        try:
            for line in fin:
                data = json.loads(line)
                datas.append(data)
        except Exception:
            raise ValueError(f"read jsonl file: {jsonl_file} failed")
    return datas


def add_jsonl_file(jsonl_file: str, data: list[dict], encoding: str = "utf-8"):
    folder_path = Path(jsonl_file).parent
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)

    with open(jsonl_file, "a", encoding=encoding) as fout:
        for json_item in data:
            fout.write(json.dumps(json_item) + "\n")


class NoMoneyException(Exception):
    """Raised when the operation cannot be completed due to insufficient funds"""

    def __init__(self, amount, message="Insufficient funds"):
        self.amount = amount
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} -> Amount required: {self.amount}"


class OutputParser:
    @classmethod
    def parse_blocks(cls, text: str):
        # 首先根据"##"将文本分割成不同的block
        blocks = text.split(MARKDOWN_TITLE_PREFIX)

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() != "":
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = block.split("\n", 1)
                # LLM可能出错，在这里做一下修正
                if block_title[-1] == ":":
                    block_title = block_title[:-1]
                block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, text: str, lang: str = "") -> str:
        pattern = rf"```{lang}.*?\s+(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise Exception
        return code

    @classmethod
    def parse_str(cls, text: str):
        text = text.split("=")[-1]
        text = text.strip().strip("'").strip('"')
        return text

    @classmethod
    def parse_file_list(cls, text: str) -> list[str]:
        # Regular expression pattern to find the tasks list.
        pattern = r"\s*(.*=.*)?(\[.*\])"

        # Extract tasks list string using regex.
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            tasks = ast.literal_eval(tasks_list_str)
        else:
            tasks = text.split("\n")
        return tasks

    @staticmethod
    def parse_python_code(text: str) -> str:
        for pattern in (
            r"(.*?```python.*?\s+)?(?P<code>.*)(```.*?)",
            r"(.*?```python.*?\s+)?(?P<code>.*)",
        ):
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue
            code = match.group("code")
            if not code:
                continue
            with contextlib.suppress(Exception):
                ast.parse(code)
                return code
        raise ValueError("Invalid python code")

    @classmethod
    def parse_data(cls, data):
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception:
                # 尝试解析list
                try:
                    content = cls.parse_file_list(text=content)
                except Exception:
                    pass
            parsed_data[block] = content
        return parsed_data

    @staticmethod
    def extract_content(text, tag="CONTENT"):
        # Use regular expression to extract content between [CONTENT] and [/CONTENT]
        extracted_content = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", text, re.DOTALL)

        if extracted_content:
            return extracted_content.group(1).strip()
        else:
            raise ValueError(f"Could not find content between [{tag}] and [/{tag}]")

    @classmethod
    def parse_data_with_mapping(cls, data, mapping):
        if "[CONTENT]" in data:
            data = cls.extract_content(text=data)
        block_dict = cls.parse_blocks(data)
        parsed_data = {}
        for block, content in block_dict.items():
            # 尝试去除code标记
            try:
                content = cls.parse_code(text=content)
            except Exception:
                pass
            typing_define = mapping.get(block, None)
            if isinstance(typing_define, tuple):
                typing = typing_define[0]
            else:
                typing = typing_define
            if (
                typing == List[str]
                or typing == List[Tuple[str, str]]
                or typing == List[List[str]]
            ):
                # 尝试解析list
                try:
                    content = cls.parse_file_list(text=content)
                except Exception:
                    pass
            # TODO: 多余的引号去除有风险，后期再解决
            # elif typing == str:
            #     # 尝试去除多余的引号
            #     try:
            #         content = cls.parse_str(text=content)
            #     except Exception:
            #         pass
            parsed_data[block] = content
        return parsed_data

    @classmethod
    def extract_struct(cls, text: str, data_type: Union[type(list), type(dict)]) -> Union[list, dict]:  # type: ignore
        """Extracts and parses a specified type of structure (dictionary or list) from the given text.
        The text only contains a list or dictionary, which may have nested structures.

        Args:
            text: The text containing the structure (dictionary or list).
            data_type: The data type to extract, can be "list" or "dict".

        Returns:
            - If extraction and parsing are successful, it returns the corresponding data structure (list or dictionary).
            - If extraction fails or parsing encounters an error, it throw an exception.

        Examples:
            >>> text = 'xxx [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}] xxx'
            >>> result_list = OutputParser.extract_struct(text, "list")
            >>> print(result_list)
            >>> # Output: [1, 2, ["a", "b", [3, 4]], {"x": 5, "y": [6, 7]}]

            >>> text = 'xxx {"x": 1, "y": {"a": 2, "b": {"c": 3}}} xxx'
            >>> result_dict = OutputParser.extract_struct(text, "dict")
            >>> print(result_dict)
            >>> # Output: {"x": 1, "y": {"a": 2, "b": {"c": 3}}}
        """
        # Find the first "[" or "{" and the last "]" or "}"
        start_index = text.find("[" if data_type is list else "{")
        end_index = text.rfind("]" if data_type is list else "}")

        if start_index != -1 and end_index != -1:
            # Extract the structure part
            structure_text = text[start_index : end_index + 1]

            try:
                # Attempt to convert the text to a Python data type using ast.literal_eval
                result = ast.literal_eval(structure_text)

                # Ensure the result matches the specified data type
                if isinstance(result, (list, dict)):
                    return result

                raise ValueError(f"The extracted structure is not a {data_type}.")

            except (ValueError, SyntaxError) as e:
                raise Exception(
                    f"Error while extracting and parsing the {data_type}: {e}"
                )
        else:
            logger.error(f"No {data_type} found in the text.")
            return [] if data_type is list else {}

    @classmethod
    def parse_xml(cls, text: str, args: list[str], json_args: list[str] = []) -> dict:
        soup = BeautifulSoup(text, "html.parser")
        res = {}
        for arg in args:
            field = soup.find(arg)
            if not field:
                raise Exception(f"Field {arg} not found in the xml")
            res[arg] = field.get_text(strip=True)
        for json_arg in json_args:
            res[json_arg] = json.loads(res[json_arg])
        return res


class CodeParser:
    @classmethod
    def parse_block(cls, block: str, text: str) -> str:
        blocks = cls.parse_blocks(text)
        for k, v in blocks.items():
            if block in k:
                return v
        return ""

    @classmethod
    def parse_blocks(cls, text: str):
        # 首先根据"##"将文本分割成不同的block
        blocks = text.split("##")

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() == "":
                continue
            if "\n" not in block:
                block_title = block
                block_content = ""
            else:
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = block.split("\n", 1)
            block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, text: str, lang: str = "", block: Optional[str] = None) -> str:
        if block:
            text = cls.parse_block(block, text)
        pattern = rf"```{lang}.*?\s+(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            logger.error(f"{pattern} not match following text:")
            logger.error(text)
            # raise Exception
            return text  # just assume original text is code
        return code

    @classmethod
    def parse_str(cls, block: str, text: str, lang: str = ""):
        code = cls.parse_code(block=block, text=text, lang=lang)
        code = code.split("=")[-1]
        code = code.strip().strip("'").strip('"')
        return code

    @classmethod
    def parse_file_list(cls, block: str, text: str, lang: str = "") -> list[str]:
        # Regular expression pattern to find the tasks list.
        code = cls.parse_code(block=block, text=text, lang=lang)
        # print(code)
        pattern = r"\s*(.*=.*)?(\[.*\])"

        # Extract tasks list string using regex.
        match = re.search(pattern, code, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            tasks = ast.literal_eval(tasks_list_str)
        else:
            raise Exception
        return tasks


def decode_image(img_url_or_b64: str) -> Image.Image:
    """decode image from url or base64 into PIL.Image"""
    if img_url_or_b64.startswith("http"):
        # image http(s) url
        resp = requests.get(img_url_or_b64)
        img = Image.open(BytesIO(resp.content))
    else:
        # image b64_json
        b64_data = re.sub("^data:image/.+;base64,", "", img_url_or_b64)
        img_data = BytesIO(base64.b64decode(b64_data))
        img = Image.open(img_data)
    return img


def log_and_reraise(retry_state: RetryCallState):
    if retry_state.outcome and hasattr(retry_state.outcome, "exception"):
        exc = retry_state.outcome.exception()
        if exc:
            logger.error(f"Retry attempts exhausted. Last exception: {exc}")
            raise exc
    logger.error("Retry attempts exhausted. No exception information available.")
    raise Exception("Retry attempts exhausted")


def general_after_log(
    i: "loguru.Logger", sec_format: str = "%0.3f"
) -> Callable[["RetryCallState"], None]:
    """
    Generates a logging function to be used after a call is retried.

    This generated function logs an error message with the outcome of the retried function call. It includes
    the name of the function, the time taken for the call in seconds (formatted according to `sec_format`),
    the number of attempts made, and the exception raised, if any.

    :param i: A Logger instance from the loguru library used to log the error message.
    :param sec_format: A string format specifier for how to format the number of seconds since the start of the call.
                       Defaults to three decimal places.
    :return: A callable that accepts a RetryCallState object and returns None. This callable logs the details
             of the retried call.
    """

    def log_it(retry_state: "RetryCallState") -> None:
        # If the function name is not known, default to "<unknown>"
        if retry_state.fn is None:
            fn_name = "<unknown>"
        else:
            # Retrieve the callable's name using a utility function
            fn_name = _utils.get_callback_name(retry_state.fn)

        # Log an error message with the function name, time since start, attempt number, and the exception
        exception_info = (
            retry_state.outcome.exception() if retry_state.outcome else None
        )
        i.error(
            f"Finished call to '{fn_name}' after {sec_format % retry_state.seconds_since_start}(s), "
            f"this was the {_utils.to_ordinal(retry_state.attempt_number)} time calling it. "
            f"exp: {exception_info}"
        )

    return log_it


@handle_exception
async def aread(filename: str | Path, encoding="utf-8") -> str:
    """Read file asynchronously."""
    if not filename or not Path(filename).exists():
        return ""
    try:
        async with aiofiles.open(str(filename), mode="r", encoding=encoding) as reader:
            content = await reader.read()
    except UnicodeDecodeError:
        async with aiofiles.open(str(filename), mode="rb") as reader:
            raw = await reader.read()
            result = chardet.detect(raw)
            detected_encoding = result["encoding"] or "utf-8"
            content = raw.decode(detected_encoding)
    return content
