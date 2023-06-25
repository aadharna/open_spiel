from typing import List, Dict, Union, Optional

from pydantic import BaseModel

class ThreadHeartbeatResponse(BaseModel):
    timestamp: str
    thread_id: int


class NodeHeartbeatResponse(BaseModel):
    timestamp: str
    hostname: str
    threads: Union[List[ThreadHeartbeatResponse],None]
    status: Optional[str] = None

class ClusterHeartbeatResponse(BaseModel):
    timestamp: str
    nodes: Dict[str,NodeHeartbeatResponse]


class GlobalHeartbeatResponse(BaseModel):
    timestamp: str
    actors: Dict[str,ClusterHeartbeatResponse]
    evaluators: Dict[str,ClusterHeartbeatResponse]

