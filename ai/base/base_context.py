import os
import json
from typing import Optional, Type, TypeVar, Union
from pathlib import Path
from abc import ABC
from pydantic import BaseModel
from utils.exceptions import handle_exception
from utils.common import aread

T = TypeVar("T", bound="BaseModel")


class BaseContext(BaseModel, ABC):
    @classmethod
    @handle_exception
    def loads(cls: Type[T], val: str) -> Optional[T]:
        i = json.loads(val)
        return cls(**i)


class Document(BaseModel):
    """
    Represents a document.
    """

    root_path: Union[str, Path] = ""
    filename: str = ""
    content: str = ""

    def get_meta(self) -> "Document":
        """Get metadata of the document.

        :return: A new Document instance with the same root path and filename.
        """

        return Document(root_path=self.root_path, filename=self.filename)

    @property
    def root_relative_path(self):
        """Get relative path from root of git repository.

        :return: relative path from root of git repository.
        """
        return os.path.join(self.root_path, self.filename)

    def __str__(self):
        return self.content

    def __repr__(self):
        return self.content

    @classmethod
    async def load(
        cls, filename: Union[str, Path], project_path: Optional[Union[str, Path]] = None
    ) -> Optional["Document"]:
        """
        Load a document from a file.

        Args:
            filename (Union[str, Path]): The path to the file to load.
            project_path (Optional[Union[str, Path]], optional): The path to the project. Defaults to None.

        Returns:
            Optional[Document]: The loaded document, or None if the file does not exist.

        """
        if not filename or not Path(filename).exists():
            return None
        content = await aread(filename=filename)
        doc = cls(content=content, filename=str(filename))
        if project_path and Path(filename).is_relative_to(project_path):
            doc.root_path = Path(filename).relative_to(project_path).parent
            doc.filename = Path(filename).name
        return doc
