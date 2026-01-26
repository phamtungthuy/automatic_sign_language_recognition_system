import os
from pydantic import BaseModel, Field, model_validator
from string import Template
from typing import Dict, Union, List, Any
from typing_extensions import Self

from utils.logs import logger
from utils.constants import ROOT_PATH

from ai.schema.message import Message


class PromptTemplateManager(BaseModel):
    templates: Dict[str, List[Message]] = Field(
        default_factory=dict,
        description="A dict from prompt template names to templates. A prompt template can be a Template instance or a chat history which is a list of dict with content as Template instance.",
    )
    templates_dir: str = ""
    template_dirs: List[str] = Field(
        default_factory=list,
        description="Additional custom directories to search for prompt templates",
    )

    @model_validator(mode="after")
    def initialize_templates(self) -> Self:
        self.templates_dir = os.path.join(ROOT_PATH, "questin", "prompts", "templates")
        self._load_templates()
        return self

    def _load_templates(self) -> None:
        """
        Load all templates from Python scripts in the templates directory.
        """
        # Load from default templates directory
        self._load_templates_from_dir(self.templates_dir, "questin.prompts.templates")

        # Load from custom templates directories
        for custom_dir in self.template_dirs:
            if os.path.exists(custom_dir):
                # Determine module path based on directory path
                relative_path = os.path.relpath(custom_dir, ROOT_PATH)
                module_path = relative_path.replace(os.sep, ".")
                self._load_templates_from_dir(custom_dir, module_path)

    def _load_templates_from_dir(
        self, templates_dir: str, templates_module: str
    ) -> None:
        """
        Load templates from a specific directory with given module path.
        """
        if not os.path.exists(templates_dir):
            if templates_dir == self.templates_dir:
                logger.error(f"Templates directory '{templates_dir}' does not exist.")
                raise FileNotFoundError(
                    f"Templates directory '{templates_dir}' does not exist."
                )
            else:
                logger.warning(
                    f"Custom templates directory '{templates_dir}' does not exist, skipping."
                )
                return

        logger.info(f"Loading templates from directory: {templates_dir}")

        for filename in os.listdir(templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]

                # Skip if template already exists (prioritize default templates directory)
                if script_name in self.templates:
                    logger.debug(
                        f"Template '{script_name}' already exists, skipping from '{templates_dir}'"
                    )
                    continue

                try:
                    module_name = f"{templates_module}.{script_name}"
                    module = __import__(module_name, fromlist=[""])

                    if not hasattr(module, "prompt_template"):
                        logger.error(
                            f"Module '{module_name}' does not define a 'prompt_template'."
                        )
                        raise AttributeError(
                            f"Module '{module_name}' does not define a 'prompt_template'."
                        )
                    prompt_template = module.prompt_template
                    logger.debug(f"Loaded template from {module_name}")
                    if isinstance(prompt_template, list) and all(
                        isinstance(item, Message) for item in prompt_template
                    ):
                        self.templates[script_name] = prompt_template
                except:
                    import traceback

                    traceback.print_exc()
                    raise TypeError(
                        f"Invalid prompt_template format in '{module_name}.py'. Must be a Template or List[Dict]."
                    )

                logger.debug(
                    f"Successfully loaded template '{script_name}' from '{module_name}.py'"
                )

    def get_template(self, name: str) -> List[Message]:
        if name not in self.templates:
            logger.error(f"Template '{name}' not found")
            raise KeyError(f"Template '{name}' not found.")
        logger.debug(f"Retrieved template '{name}'.")

        return self.templates[name]

    def render(self, name: str, **kwargs) -> List[Message]:
        template = self.get_template(name)
        try:
            rendered_list = [
                Message(
                    role=item.role,
                    content=Template(item.content).substitute(**kwargs),
                    cause_by=item.cause_by,
                    sent_from=item.sent_from,
                    send_to=item.send_to,
                    metadata=item.metadata,
                )
                for item in template
            ]
            return rendered_list
        except KeyError as e:
            logger.error(f"Missing variable in chat history template '{name}': {e}")
            raise ValueError(f"Missing variable in chat history template '{name}': {e}")
        return rendered_list
