from typing import Optional, Dict

from pydantic import BaseModel
from reverb.reverb_types import SpecNest


class Table(BaseModel):
    name: str
#    sampler_options: schema_pb2.KeyDistributionOptions
#    remover_options: schema_pb2.KeyDistributionOptions
    max_size: int
    max_times_sampled: int
#    rate_limiter_info: schema_pb2.RateLimiterInfo
#    signature: Optional[SpecNest]
    current_size: int
    num_episodes: int
    num_deleted_episodes: int
    num_unique_samples: int
#    table_worker_time: schema_pb2.TableWorkerTime

    @staticmethod
    def from_table_row(table_row) -> 'Table':
        return Table(
            name=table_row.name,
            max_size=table_row.max_size,
            max_times_sampled=table_row.max_times_sampled,
#            signature=table_row.signature,
            current_size=table_row.current_size,
            num_episodes=table_row.num_episodes,
            num_deleted_episodes=table_row.num_deleted_episodes,
            num_unique_samples=table_row.num_unique_samples,
        )

    @staticmethod
    def from_server_info(server_info) -> Dict[str,'Table']:
        return {table_name: Table.from_table_row(table) for table_name,table in server_info.items()}


