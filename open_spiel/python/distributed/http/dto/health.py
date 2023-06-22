from typing import List

from pydantic import BaseModel


class ThreadHealth(BaseModel):
    node: str
    status: int

class HealthResponse(BaseModel):
    timestamp: str
    status: int
    threads: List[ThreadHealth]

