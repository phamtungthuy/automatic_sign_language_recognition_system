import json
from json import JSONDecodeError
import uuid
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    PrivateAttr,
    ConfigDict,
    create_model,
)
from typing import Any, Optional, Dict, Union, List
import asyncio
from asyncio import wait_for, Queue, QueueEmpty

from utils.constants import (
    MESSAGE_ROUTE_TO_ALL,
    AGENT,
    MESSAGE_ROUTE_CAUSE_BY,
    MESSAGE_ROUTE_FROM,
    MESSAGE_ROUTE_TO,
)
from utils.logs import logger
from utils.common import (
    any_to_str,
    any_to_str_set,
    CodeParser,
    import_class,
)
from utils.exceptions import handle_exception

from ai.utils.serialize import (
    actionoutout_schema_to_mapping,
    actionoutput_mapping_to_str,
    actionoutput_str_to_mapping,
)


class Resource(BaseModel):
    """Used by `Message`.`parse_resources`"""

    resource_type: str  # the type of resource
    value: str  # a string type of resource content
    description: str  # explanation


class Message(BaseModel):
    """list[<role>: <content>]"""

    id: str = Field(
        default="", validate_default=True
    )  # According to Section 2.2.3.1.1 of RFC 135
    content: str  # natural language for user or agent
    instruct_content: Optional[BaseModel] = Field(default=None, validate_default=True)
    role: str = "user"  # system / user / assistant
    cause_by: str = Field(default="", validate_default=True)
    sent_from: str = Field(default="", validate_default=True)
    send_to: set[str] = Field(default={MESSAGE_ROUTE_TO_ALL}, validate_default=True)
    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )  # metadata for `content` and `instruct_content`

    @field_validator("id", mode="before")
    @classmethod
    def check_id(cls, id: str) -> str:
        return id if id else uuid.uuid4().hex

    @field_validator("instruct_content", mode="before")
    @classmethod
    def check_instruct_content(cls, ic: Any) -> BaseModel:
        if ic and isinstance(ic, dict) and "class" in ic:
            if "mapping" in ic:
                # compatible with custom-defined ActionOutput
                mapping = actionoutput_str_to_mapping(ic["mapping"])
                actionnode_class = import_class(
                    "ActionNode", "metagpt.actions.action_node"
                )  # avoid circular import
                ic_obj = actionnode_class.create_model_class(
                    class_name=ic["class"], mapping=mapping
                )
            elif "module" in ic:
                # subclasses of BaseModel
                ic_obj = import_class(ic["class"], ic["module"])
            else:
                raise KeyError(
                    "missing required key to init Message.instruct_content from dict"
                )
            ic = ic_obj(**ic["value"])
        return ic

    @field_validator("cause_by", mode="before")
    @classmethod
    def check_cause_by(cls, cause_by: Any) -> str:
        return any_to_str(
            cause_by
            if cause_by
            else import_class("UserRequirement", "questin.actions.user_requirement")
        )

    @field_validator("sent_from", mode="before")
    @classmethod
    def check_sent_from(cls, sent_from: Any) -> str:
        return any_to_str(sent_from if sent_from else "")

    @field_validator("send_to", mode="before")
    @classmethod
    def check_send_to(cls, send_to: Any) -> set:
        return any_to_str_set(send_to if send_to else {MESSAGE_ROUTE_TO_ALL})

    @field_serializer("send_to", mode="plain")
    def ser_send_to(self, send_to: set) -> list:
        return list(send_to)

    @field_serializer("instruct_content", mode="plain")
    def ser_instruct_content(self, ic: BaseModel) -> Union[dict, None]:
        ic_dict = None
        if ic:
            # compatible with custom-defined ActionOutput
            schema = ic.model_json_schema()
            ic_type = str(type(ic))
            if "<class 'metagpt.actions.action_node" in ic_type:
                # instruct_content from AutoNode.create_model_class, for now, it's single level structure.
                mapping = actionoutout_schema_to_mapping(schema)
                mapping = actionoutput_mapping_to_str(mapping)

                ic_dict = {
                    "class": schema["title"],
                    "mapping": mapping,
                    "value": ic.model_dump(),
                }
            else:
                # due to instruct_content can be assigned by subclasses of BaseModel
                ic_dict = {
                    "class": schema["title"],
                    "module": ic.__module__,
                    "value": ic.model_dump(),
                }
        return ic_dict

    def __init__(self, content: str = "", **data: Any):
        data["content"] = data.get("content", content)
        super().__init__(**data)

    def __setattr__(self, key, val):
        """Override `@property.setter`, convert non-string parameters into string parameters."""
        if key == MESSAGE_ROUTE_CAUSE_BY:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_FROM:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_TO:
            new_val = any_to_str_set(val)
        else:
            new_val = val
        super().__setattr__(key, new_val)

    def __str__(self):
        # prefix = '-'.join([self.role, str(self.cause_by)])
        # if self.instruct_content:
        #     return f"{self.role}: {self.instruct_content.model_dump()}"
        # return f"{self.role}: {self.content}"
        if self.instruct_content:
            return str(
                {"role": self.role, "content": self.instruct_content.model_dump()}
            )
        return str({"role": self.role, "content": self.content})

    def __repr__(self):
        return self.__str__()

    def rag_key(self) -> str:
        """For search"""
        return self.content

    def to_dict(self) -> dict:
        """Return a dict containing `role` and `content` for the LLM call.l"""
        return {"role": self.role, "content": self.content}

    def dump(self) -> str:
        """Convert the object to json string"""
        return self.model_dump_json(exclude_none=True, warnings=False)

    @staticmethod
    @handle_exception(exception_type=JSONDecodeError, default_return=None)
    def load(val):
        """Convert the json string to object."""

        try:
            m = json.loads(val)
            id = m.get("id")
            if "id" in m:
                del m["id"]
            msg = Message(**m)
            if id:
                msg.id = id
            return msg
        except JSONDecodeError as err:
            logger.error(f"parse json failed: {val}, error:{err}")
        return None

    async def parse_resources(
        self, llm: "BaseLLM", key_descriptions: Dict[str, str] = None
    ) -> Dict:
        """
        `parse_resources` corresponds to the in-context adaptation capability of the input of the atomic action,
        which will be migrated to the context builder later.

        Args:
            llm (BaseLLM): The instance of the BaseLLM class.
            key_descriptions (Dict[str, str], optional): A dictionary containing descriptions for each key,
                if provided. Defaults to None.

        Returns:
            Dict: A dictionary containing parsed resources.

        """
        if not self.content:
            return {}
        content = f"## Original Requirement\n```text\n{self.content}\n```\n"
        return_format = (
            "Return a markdown JSON object with:\n"
            '- a "resources" key contain a list of objects. Each object with:\n'
            '  - a "resource_type" key explain the type of resource;\n'
            '  - a "value" key containing a string type of resource content;\n'
            '  - a "description" key explaining why;\n'
        )
        key_descriptions = key_descriptions or {}
        for k, v in key_descriptions.items():
            return_format += f'- a "{k}" key containing {v};\n'
        return_format += '- a "reason" key explaining why;\n'
        instructions = [
            'Lists all the resources contained in the "Original Requirement".',
            return_format,
        ]
        rsp = await llm.aask(msg=content, system_msgs=instructions)
        json_data = CodeParser.parse_code(text=rsp, lang="json")
        m = json.loads(json_data)
        m["resources"] = [Resource(**i) for i in m.get("resources", [])]
        return m

    def add_metadata(self, key: str, value: str):
        self.metadata[key] = value

    @staticmethod
    def create_instruct_value(kvs: Dict[str, Any], class_name: str = "") -> BaseModel:
        """
        Dynamically creates a Pydantic BaseModel subclass based on a given dictionary.

        Parameters:
        - data: A dictionary from which to create the BaseModel subclass.

        Returns:
        - A Pydantic BaseModel subclass instance populated with the given data.
        """
        if not class_name:
            class_name = "DM" + uuid.uuid4().hex[0:8]
        dynamic_class = create_model(
            class_name, **{key: (value.__class__, ...) for key, value in kvs.items()}
        )
        return dynamic_class.model_validate(kvs)

    def is_user_message(self) -> bool:
        return self.role == "user"

    def is_ai_message(self) -> bool:
        return self.role == "assistant"


class UserMessage(Message):
    """便于支持OpenAI的消息
    Facilitate support for OpenAI messages
    """

    def __init__(self, content: str, **kwargs):
        kwargs.pop("role", None)
        super().__init__(content=content, role="user", **kwargs)


class SystemMessage(Message):
    """便于支持OpenAI的消息
    Facilitate support for OpenAI messages
    """

    def __init__(self, content: str, **kwargs):
        kwargs.pop("role", None)
        super().__init__(content=content, role="system", **kwargs)


class AIMessage(Message):
    """便于支持OpenAI的消息
    Facilitate support for OpenAI messages
    """

    def __init__(self, content: str, **kwargs):
        kwargs.pop("role", None)
        super().__init__(content=content, role="assistant", **kwargs)

    def with_agent(self, name: str):
        self.add_metadata(key=AGENT, value=name)
        return self

    @property
    def agent(self) -> str:
        return self.metadata.get(AGENT, "")


class MessageQueue(BaseModel):
    """Message queue which supports asynchronous updates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _queue: Queue = PrivateAttr(default_factory=Queue)

    def pop(self) -> Message | None:
        """Pop one message from the queue."""
        try:
            item = self._queue.get_nowait()
            if item:
                self._queue.task_done()
            return item
        except QueueEmpty:
            return None

    def pop_all(self) -> List[Message]:
        """Pop all messages from the queue."""
        ret = []
        while True:
            msg = self.pop()
            if not msg:
                break
            ret.append(msg)
        return ret

    def push(self, msg: Message):
        """Push a message into the queue."""
        self._queue.put_nowait(msg)

    def empty(self):
        """Return true if the queue is empty."""
        return self._queue.empty()

    async def dump(self) -> str:
        """Convert the `MessageQueue` object to a json string."""
        if self.empty():
            return "[]"

        lst = []
        msgs = []
        try:
            while True:
                item = await wait_for(self._queue.get(), timeout=1.0)
                if item is None:
                    break
                msgs.append(item)
                lst.append(item.dump())
                self._queue.task_done()
        except asyncio.TimeoutError:
            logger.debug("Queue is empty, exiting...")
        finally:
            for m in msgs:
                self._queue.put_nowait(m)
        return json.dumps(lst, ensure_ascii=False)

    @staticmethod
    def load(data) -> "MessageQueue":
        """Convert the json string to the `MessageQueue` object."""
        queue = MessageQueue()
        try:
            lst = json.loads(data)
            for i in lst:
                msg = Message.load(i)
                queue.push(msg)
        except JSONDecodeError as e:
            logger.warning(f"JSON load failed: {data}, error:{e}")

        return queue
