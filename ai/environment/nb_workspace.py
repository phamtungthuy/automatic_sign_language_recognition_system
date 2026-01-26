import re
import base64
import asyncio
import pickle
from datetime import datetime
from typing import Literal, Tuple, Optional, Any, cast, Dict
from pydantic import BaseModel

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output, output_from_msg
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionComplete, CellTimeoutError, DeadKernelError
from nbclient.util import ensure_async
from rich.box import MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from pydantic import Field

from utils.logs import logger

from ai.utils.report import NotebookReporter
from ai.schema.memory import Memory
from ai.schema.context_mixin import ContextMixin


INSTALL_KEEPLEN = 500
INI_CODE = """import warnings
import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
warnings.filterwarnings('ignore')"""


class RealtimeOutputNotebookClient(NotebookClient):
    """Realtime output of Notebook execution."""

    def __init__(self, *args, notebook_reporter=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.notebook_reporter = notebook_reporter or NotebookReporter()

    async def _async_poll_output_msg(
        self, parent_msg_id: str, cell: NotebookNode, cell_index: int
    ) -> None:
        """Implement a feature to enable sending messages."""
        assert self.kc is not None
        while True:
            msg = await ensure_async(self.kc.iopub_channel.get_msg(timeout=None))
            await self._send_msg(msg)

            if msg["parent_header"].get("msg_id") == parent_msg_id:
                try:
                    # Will raise CellExecutionComplete when completed
                    self.process_message(msg, cell, cell_index)
                except CellExecutionComplete:
                    return

    async def _send_msg(self, msg: dict):
        msg_type = msg.get("header", {}).get("msg_type")
        if msg_type not in ["stream", "error", "execute_result"]:
            return

        await self.notebook_reporter.async_report(output_from_msg(msg), "content")


class NbCellRecord:
    """Record the code and output of the notebook"""

    def __init__(
        self,
        code: str,
        language: str,
        output: str,
        success: bool,
        timestamp: Optional[datetime] = None,
        metadata: dict = {},
    ):
        self.code = code
        self.language = language
        self.output = output
        self.success = success
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata
        self.cell_index: Optional[int] = None

    def __str__(self):
        status = "✅" if self.success else "❌"
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {status} {self.language.upper()}"

    def format_history(
        self,
        include_output: bool = True,
        max_code_len: int = 1000,
        max_output_len: int = 500,
    ):
        status = "✅ SUCCESS" if self.success else "❌ ERROR"
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        code_display = self.code
        if len(code_display) > max_code_len:
            code_display = code_display[:max_code_len] + "..."
        result = f"""
Cell {self.cell_index}:
Task type: {self.metadata["task_type"]}
Task {self.metadata["task_id"]}: {self.metadata["task_problem"]}

{'='*60}
📅 {time_str} | {status} | {self.language.upper()}
{'='*60}
CODE:
{code_display}"""

        if include_output and self.output:
            output_display = self.output
            if len(output_display) > max_output_len:
                output_display = output_display[:max_output_len] + "..."

            result += f"""
----OUTPUT----
{output_display}"""

        return result


class NbCodeWorkspace(ContextMixin, BaseModel):
    nb: NotebookNode = Field(default_factory=lambda: nbformat.v4.new_notebook())
    nb_client: Optional[RealtimeOutputNotebookClient] = None
    console: Console = Field(default_factory=Console)
    interaction: str = Field(default="")
    timeout: int = 600
    enable_rollback: bool = Field(default=False)
    history: list[NbCellRecord] = Field(default_factory=list)
    max_history_size: int = 100
    memory: Memory = Field(default_factory=Memory)

    def __init__(self, nb=nbformat.v4.new_notebook(), timeout=600, **kwargs):
        super().__init__(**kwargs)
        self.nb = nb
        self.timeout = timeout
        self.console = Console()
        self.interaction = "ipython" if self.is_ipython() else "terminal"
        self.reporter = NotebookReporter()
        self.set_nb_client()
        self.init_called = False

    async def init_code(self):
        if not self.init_called:
            await self.run(INI_CODE)
            self.init_called = True

    def set_nb_client(self):
        self.nb_client = RealtimeOutputNotebookClient(
            self.nb,
            timeout=self.timeout,
            resources={"metadata": {"path": self.config.workspace.path}},
            notebook_reporter=self.reporter,
            coalesce_streams=True,
        )

    async def build(self):
        if self.nb_client is None:
            return
        if self.nb_client.kc is None or not await self.nb_client.kc._async_is_alive():
            self.nb_client.create_kernel_manager()
            self.nb_client.start_new_kernel()
            self.nb_client.start_new_kernel_client()

    async def terminate(self):
        """kill NotebookClient"""
        if self.nb_client is None:
            return
        if self.nb_client.km is not None and await self.nb_client.km._async_is_alive():
            await self.nb_client.km._async_shutdown_kernel(now=True)
            await self.nb_client.km._async_cleanup_resources()

            if self.nb_client.kc is not None:
                channels = [
                    self.nb_client.kc.stdin_channel,  # The channel for handling standard input to the kernel.
                    self.nb_client.kc.hb_channel,  # The channel for heartbeat communication between the kernel and client.
                    self.nb_client.kc.control_channel,  # The channel for controlling the kernel.
                ]

                # Stops all the running channels for this kernel
                for channel in channels:
                    if channel.is_alive():
                        channel.stop()

            self.nb_client.kc = None
            self.nb_client.km = None

    async def reset(self):
        """reset NotebookClient"""
        await self.terminate()

        # sleep 1s to wait for the kernel to be cleaned up completely
        await asyncio.sleep(1)
        await self.build()
        self.set_nb_client()

    def add_to_history(
        self, code: str, language: str, output: str, success: bool, metadata: dict = {}
    ):
        record = NbCellRecord(
            code=code, language=language, output=output, success=success
        )
        record.cell_index = len(self.nb.cells)
        record.metadata = metadata
        self.history.append(record)

        if len(self.history) > self.max_history_size:
            self.history.pop(0)

    def get_history(
        self,
        last_n: Optional[int] = None,
        include_output: bool = True,
        filter_success: Optional[bool] = None,
        filter_language: Optional[str] = None,
        filter_task_ids: list = [],
    ):
        # Filter records
        filtered_records = self.history

        if filter_success is not None:
            filtered_records = [
                r for r in filtered_records if r.success == filter_success
            ]

        if filter_language:
            filtered_records = [
                r for r in filtered_records if r.language == filter_language
            ]

        if len(filter_task_ids) > 0:
            filtered_records = [
                r for r in filtered_records if r.metadata["task_id"] in filter_task_ids
            ]

        # Limit records
        if last_n:
            filtered_records = filtered_records[-last_n:]
        result = f"""
🚀 EXECUTION HISTORY ({len(filtered_records)} records)
{'='*70}"""
        for i, record in enumerate(filtered_records, 1):
            result += f"\n{i}. {record.format_history(include_output)}"

        result += f"\n{'='*70}"
        return result

    async def create_checkpoint(self) -> Dict[str, Any]:
        if not self.enable_rollback:
            return {}
        checkpoint_code = """
import pickle
import base64

# Get all variables in global namespace
_checkpoint_vars = {}
for name, value in globals().copy().items():
    if not name.startswith('_') and not callable(value) and not name in ['In', 'Out']:
        try:
            # Try to serialize to ensure it can be backed up
            pickle.dumps(value)
            _checkpoint_vars[name] = value
        except:
            # Skip variables that cannot be serialized
            pass

# Convert to base64 string to transfer
_checkpoint_pickle = pickle.dumps(_checkpoint_vars)
_checkpoint_b64 = base64.b64encode(_checkpoint_pickle).decode('utf-8')
print(f"CHECKPOINT:{_checkpoint_b64}")
"""
        self.add_code_cell(checkpoint_code)
        cell_index = len(self.nb.cells) - 1
        await self.build()

        success, output = await self.run_cell(self.nb.cells[-1], cell_index)

        # Parse checkpoint from output
        checkpoint_data = {}
        if success and "CHECKPOINT:" in output:
            try:
                import base64
                import pickle

                checkpoint_b64 = output.split("CHECKPOINT:")[1].strip()
                checkpoint_pickle = base64.b64decode(checkpoint_b64)
                checkpoint_data = pickle.loads(checkpoint_pickle)

                print(f"✅ Đã tạo checkpoint với {len(checkpoint_data)} variables")
            except Exception as e:
                print(f"⚠️ Lỗi khi tạo checkpoint: {e}")

        # Remove checkpoint cell from notebook
        self.nb.cells.pop()

        return checkpoint_data

    async def restore_checkpoint(self, checkpoint_data: Dict[str, Any]):
        if not checkpoint_data or not self.enable_rollback:
            return
        restore_code = f"""
import pickle
import base64

# Decode checkpoint data  
_checkpoint_b64 = "{base64.b64encode(pickle.dumps(checkpoint_data)).decode('utf-8')}"
_checkpoint_pickle = base64.b64decode(_checkpoint_b64)
_restored_vars = pickle.loads(_checkpoint_pickle)

# Clear current variables (except built-ins)
_vars_to_delete = []
for name in list(globals().keys()):
    if not name.startswith('_') and not callable(globals()[name]) and name not in ['In', 'Out']:
        _vars_to_delete.append(name)

for name in _vars_to_delete:
    try:
        del globals()[name]
    except:
        pass

# Restore variables from checkpoint
globals().update(_restored_vars)
print(f"✅ Restore {{len(_restored_vars)}} variables from checkpoint")

# Clean up temp variables
del _checkpoint_b64, _checkpoint_pickle, _restored_vars, _vars_to_delete
"""
        # Run restore code
        self.add_code_cell(restore_code)
        cell_index = len(self.nb.cells) - 1
        await self.build()

        success, output = await self.run_cell(self.nb.cells[-1], cell_index)
        if success:
            print(f"✅ Restore successfully: {output}")
        else:
            print(f"❌ Error when restore: {output}")

        # Remove restore cell from notebook
        self.nb.cells.pop()

    def add_code_cell(self, code: str):
        self.nb.cells.append(new_code_cell(source=code))

    def add_markdown_cell(self, markdown: str):
        self.nb.cells.append(new_markdown_cell(source=markdown))

    def remove_last_cell(self):
        """Remove last cell from notebook"""
        if len(self.nb.cells) > 0:
            removed_cell = self.nb.cells.pop()
            print(f"🗑️ Remove last cell: {removed_cell.source[:50]}...")
            return removed_cell
        return None

    def remove_cell_by_index(self, index: int):
        """Remove cell by index"""
        if 0 <= index < len(self.nb.cells):
            removed_cell = self.nb.cells.pop(index)
            print(f"🗑️ Remove cell {index}: {removed_cell.source[:50]}...")
            return removed_cell
        return None

    def _display(self, code: str, language: Literal["python", "markdown"] = "python"):
        if language == "python":
            syntax = Syntax(code, "python", theme="paraiso-dark", line_numbers=True)
            self.console.print(syntax)
        elif language == "markdown":
            display_markdown(code)
        else:
            raise ValueError(f"Only support for python, markdown, but got {language}")

    def add_output_to_cell(self, cell: NotebookNode, output: str):
        """add outputs of code execution to notebook cell."""
        if "outputs" not in cell:
            cell["outputs"] = []
        else:
            cell["outputs"].append(
                new_output(output_type="stream", name="stdout", text=str(output))
            )

    def parse_outputs(
        self, outputs: list[str], keep_len: int = 5000
    ) -> Tuple[bool, str]:
        """Parses the outputs received from notebook execution."""
        assert isinstance(outputs, list)
        parsed_output, is_success = [], True
        for i, output in enumerate(outputs):
            output_dict = cast(dict[str, Any], output)
            output_text = ""
            if output_dict["output_type"] == "stream" and not any(
                tag in output_dict["text"]
                for tag in [
                    "| INFO     | metagpt",
                    "| ERROR    | metagpt",
                    "| WARNING  | metagpt",
                    "DEBUG",
                ]
            ):
                output_text = output_dict["text"]
            elif output_dict["output_type"] == "display_data":
                if "image/png" in output_dict["data"]:
                    interaction_type = (
                        "ipython" if self.interaction == "ipython" else None
                    )
                    self.show_bytes_figure(
                        output_dict["data"]["image/png"], interaction_type
                    )
                else:
                    logger.info(
                        f"{i}th output['data'] from nbclient outputs dont have image/png, continue next output ..."
                    )
            elif output_dict["output_type"] == "execute_result":
                output_text = output_dict["data"]["text/plain"]
            elif output_dict["output_type"] == "error":
                output_text, is_success = "\n".join(output_dict["traceback"]), False

            # handle coroutines that are not executed asynchronously
            if output_text.strip().startswith("<coroutine object"):
                output_text = "Executed code failed, you need use key word 'await' to run a async code."
                is_success = False

            output_text = remove_escape_and_color_codes(output_text)
            if is_success:
                output_text = remove_log_and_warning_lines(output_text)
            # The useful information of the exception is at the end,
            # the useful information of normal output is at the begining.
            if "<!DOCTYPE html>" not in output_text:
                output_text = (
                    output_text[:keep_len] if is_success else output_text[-keep_len:]
                )

            parsed_output.append(output_text)
        return is_success, ",".join(parsed_output)

    def show_bytes_figure(
        self, image_base64: str, interaction_type: Literal["ipython", None]
    ):
        image_bytes = base64.b64decode(image_base64)
        if interaction_type == "ipython":
            from IPython.display import Image, display

            display(Image(data=image_bytes))
        else:
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            image.show()

    def is_ipython(self) -> bool:
        try:
            # 如果在Jupyter Notebook中运行，__file__ 变量不存在
            from IPython.core.getipython import get_ipython

            ipython = get_ipython()
            if ipython is not None and "IPKernelApp" in ipython.config:
                return True
            else:
                return False
        except NameError:
            return False

    async def run_cell(self, cell: NotebookNode, cell_index: int) -> Tuple[bool, str]:
        """set timeout for run code.
        returns the success or failure of the cell execution, and an optional error message.
        """
        await self.reporter.async_report(cell, "content")

        if self.nb_client is None:
            return False, "NotebookClient not initialized"

        try:
            await self.nb_client.async_execute_cell(cell, cell_index)
            return self.parse_outputs(self.nb.cells[-1].outputs)
        except CellTimeoutError:
            assert self.nb_client.km is not None
            self.nb_client.km.interrupt_kernel()
            await asyncio.sleep(1)
            error_msg = "Cell execution timed out: Execution exceeded the time limit and was stopped; consider optimizing your code for better performance."
            return False, error_msg
        except DeadKernelError:
            await self.reset()
            return False, "DeadKernelError"
        except Exception:
            return self.parse_outputs(self.nb.cells[-1].outputs)

    async def run(
        self, code: str, language: Literal["python", "markdown"] = "python"
    ) -> Tuple[str, bool]:
        """
        return the output of code execution, and a success indicator (bool) of code execution.
        """
        self._display(code, language)

        async with self.reporter:
            if language == "python":
                # add code to the notebook
                self.add_code_cell(code=code)

                # build code executor
                await self.build()

                # run code
                cell_index = len(self.nb.cells) - 1
                success, outputs = await self.run_cell(self.nb.cells[-1], cell_index)

                if "!pip" in code:
                    success = False
                    outputs = outputs[-INSTALL_KEEPLEN:]
                elif "git clone" in code:
                    outputs = (
                        outputs[:INSTALL_KEEPLEN] + "..." + outputs[-INSTALL_KEEPLEN:]
                    )

            elif language == "markdown":
                # add markdown content to markdown cell in a notebook.
                self.add_markdown_cell(code)
                # return True, beacuse there is no execution failure for markdown cell.
                outputs, success = code, True
            else:
                raise ValueError(
                    f"Only support for language: python, markdown, but got {language}, "
                )

            file_path = self.config.workspace.path / "code.ipynb"
            nbformat.write(self.nb, file_path)
            await self.reporter.async_report(file_path, "path")

            return outputs, success

    async def safe_run(
        self,
        code: str,
        language: Literal["python", "markdown"] = "python",
        metadata: dict = {},
    ) -> Tuple[str, bool]:
        """
        Run code safely with rollback when error
        """
        if language == "markdown":
            # Markdown does not need rollback
            output, success = await self.run(code, language)
            self.add_to_history(code, language, output, success, metadata)
            return output, success

        print(f"🔄 Create checkpoint before running code...")

        # Create checkpoint before running
        checkpoint = await self.create_checkpoint()

        try:
            # Run code
            output, success = await self.run(code, language)
            output = output.replace(
                ",from .autonotebook import tqdm as notebook_tqdm", ""
            )
            if success:
                # self.add_to_history(code, language, output, success, metadata)
                print(output)
                print(f"✅ Code run successfully!")
                return output, success
            else:
                print(output)
                print(f"❌ Code error, rollback...")

                # Remove error cell
                self.remove_last_cell()

                # Restore to checkpoint
                await self.restore_checkpoint(checkpoint)

                return f"Code error and rollback. Original error: {output}", False

        except Exception as e:
            print(f"❌ Exception: {e}")

            # Add exception to history
            # self.add_to_history(code, language, str(e), False, metadata)

            # Remove error cell if any
            if len(self.nb.cells) > 0:
                self.remove_last_cell()

            # Restore to checkpoint
            await self.restore_checkpoint(checkpoint)

            return f"Exception and rollback: {str(e)}", False


def remove_log_and_warning_lines(input_str: str) -> str:

    delete_lines = ["[warning]", "warning:", "[cv]", "[info]"]
    result = "\n".join(
        [
            line
            for line in input_str.split("\n")
            if not any(dl in line.lower() for dl in delete_lines)
        ]
    ).strip()
    return result


def remove_escape_and_color_codes(input_str: str):
    # 使用正则表达式去除jupyter notebook输出结果中的转义字符和颜色代码
    # Use regular expressions to get rid of escape characters and color codes in jupyter notebook output.
    pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
    result = pattern.sub("", input_str)
    return result


def display_markdown(content: str):
    # Use regular expressions to match blocks of code one by one.
    matches = re.finditer(r"```(.+?)```", content, re.DOTALL)
    start_index = 0
    content_panels = []
    # Set the text background color and text color.
    style = "black on white"
    # Print the matching text and code one by one.
    for match in matches:
        text_content = content[start_index : match.start()].strip()
        code_content = match.group(0).strip()[3:-3]  # Remove triple backticks

        if text_content:
            content_panels.append(
                Panel(Markdown(text_content), style=style, box=MINIMAL)
            )

        if code_content:
            content_panels.append(
                Panel(Markdown(f"```{code_content}"), style=style, box=MINIMAL)
            )
        start_index = match.end()

    # Print remaining text (if any).
    remaining_text = content[start_index:].strip()
    if remaining_text:
        content_panels.append(Panel(Markdown(remaining_text), style=style, box=MINIMAL))

    # Display all panels in Live mode.
    with Live(
        auto_refresh=False, console=Console(), vertical_overflow="visible"
    ) as live:
        live.update(Group(*content_panels))
        live.refresh()
