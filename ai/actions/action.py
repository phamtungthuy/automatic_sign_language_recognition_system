from typing import Any, Optional
from pydantic import BaseModel, model_validator, ConfigDict
from provider.llm_provider_registry import create_llm_instance
from configs.models_config import ModelsConfig

from ai.schema.context_mixin import ContextMixin
from ai.schema.context import Context


class Action(ContextMixin, Context, BaseModel):
    name: str = ""
    desc: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_name_or_type: Optional[str] = None

    @model_validator(mode="after")
    @classmethod
    def _update_private_llm(cls, data: Any) -> Any:
        print(data)
        print(f"Action: {data.name} Using model: {data.llm_name_or_type}")
        try:
            config = ModelsConfig.default().get(data.llm_name_or_type)
        except:
            config = None
        if config:
            llm = create_llm_instance(config)
            llm.cost_manager = data.llm.cost_manager
            data.llm = llm
        return data

    def set_prefix(self, prefix):
        """Set prefix for later usage"""
        self.prefix = prefix
        self.llm.system_prompt = prefix
        return self

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    async def _aask(self, prompt: str, system_msgs: Optional[list[str]] = None) -> str:
        return await self.llm.aask(prompt, system_msgs)

    async def run(self, *args, **kwargs):
        """Run action"""
        raise NotImplementedError("The run method should be implemented in a subclass.")


async def main():
    action = Action()
    print(await action._aask("Hello, world!"))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
