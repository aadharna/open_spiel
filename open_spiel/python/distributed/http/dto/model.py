from typing import Optional

from pydantic import BaseModel


class ModelPath(BaseModel):
    path: str
    hash: Optional[str] = None