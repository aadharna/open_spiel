from typing import Optional

from pydantic import BaseModel


class Error(BaseModel):
    status: int
    message: str
    detail: Optional[str]