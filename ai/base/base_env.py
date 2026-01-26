from abc import abstractmethod
from typing import Any, Optional
from pydantic import BaseModel

from ai.base.base_serializer import BaseSerialization
from ai.base.base_env_space import BaseEnvObsParams, BaseEnvAction


class BaseEnvironment(BaseSerialization, BaseModel):
    """Base environment"""

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Implement this to get init observation"""

    @abstractmethod
    def observe(self, obs_params: Optional[BaseEnvObsParams] = None) -> Any:
        """Implement this if you want to get partial observation from the env"""

    @abstractmethod
    def step(
        self, action: BaseEnvAction
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Implement this to feed a action and then get new observation from the env"""

    @abstractmethod
    def publish_message(self, message: "Message", peekable: bool = True) -> bool:  # type: ignore
        """Distribute the message to the recipients."""

    @abstractmethod
    async def run(self, k=1):
        """Process all task at once"""

    @abstractmethod
    def set_addresses(self, obj, addresses):
        """Set the addresses of the object"""

    @abstractmethod
    def set_task_info(self, problem_desc: str, input_desc: str, output_desc: str):
        """Set the task info"""
