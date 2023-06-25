from pydantic import BaseModel


class Discovery(BaseModel):
    hostname: str


