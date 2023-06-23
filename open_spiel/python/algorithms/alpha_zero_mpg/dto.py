import abc
import math
import time
from typing import List, Dict, TypedDict, Union

import numpy as np
from open_spiel.python.algorithms.alpha_zero_mpg.multiprocess.orchestrator import ProcessesTrajectoryCollector
from open_spiel.python.utils import spawn, stats

import open_spiel.python.algorithms.alpha_zero_mpg.utils
from . import utils, model as model_lib
import random
import reverb
from .services import actor
from .multiprocess import queue as queue_lib


class ReplayBufferAdapter(abc.ABC):

    def __init__(self):
        pass

    def sample(self, n):
        if type(n) == float:
            if n < 0:
                raise ValueError(f"The fraction cannot be negative")
            if n <= 1:
                return self.sample_by_fraction(n)
        return self.sample_by_numbers(n)

    @abc.abstractmethod
    def sample_by_fraction(self, p):
        pass

    @abc.abstractmethod
    def sample_by_numbers(self, n):
        pass

    @abc.abstractmethod
    def length(self) -> int:
        pass

    def update(self):
        pass

    @property
    def analysis_data(self) -> Dict:
        self.update()
        return {}

    def __len__(self) -> int:
        return self.length()

    def __bool__(self) -> bool:
        return self.length() == 0

    def dataset(self):
        raise NotImplementedError("The dataset method is not implemented for this replay buffer")

    @property
    def supports_dataset(self):
        return False

    def max_length(self):
        return -1


class QueuesReplayBuffer(ReplayBufferAdapter):

    def __init__(self, replay_buffer_size: int, replay_buffer_reuse: int, actors: List[spawn.Process],
                 max_game_length: int, stage_count=None):
        super().__init__()
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer_reuse = replay_buffer_reuse
        self.replay_buffer = utils.Buffer(replay_buffer_size)
        self.trajectory_collector = ProcessesTrajectoryCollector(actors, stage_count=stage_count,
                                                                 max_game_length=max_game_length,
                                                                 learn_rate=self.learn_rate)
        self.current_analysis_data: Union[ProcessesTrajectoryCollector.CollectedMetadata, None] = None

    def update(self):
        train_data, self.current_analysis_data = self.trajectory_collector.collect()
        self.replay_buffer.extend(train_data)

    @property
    def analysis_data(self) -> Dict:
        return self.current_analysis_data

    # def sample_by_numbers(self,n):

    @property
    def learn_rate(self):
        return self.replay_buffer_size // self.replay_buffer_reuse

    def sample_by_numbers(self, n):
        return random.sample(self.replay_buffer.data, n)

    def sample_by_fraction(self, p):
        return random.sample(self.replay_buffer.data, math.ceil(p * len(self.replay_buffer)))

    def length(self) -> int:
        return len(self.replay_buffer)

    def max_length(self):
        return self.replay_buffer_size


class GrpcReplayBuffer(ReplayBufferAdapter):
    def __init__(self, address: str, port: int, table: str):
        super().__init__()
        self.address = address
        self.port = port
        self.table = table
        self.client = reverb.Client(f"{self.address}:{self.port}")

    pass

    def update(self):
        pass

    @property
    def learn_rate(self):
        raise NotImplementedError("The learn rate is not implemented for this replay buffer")

    def max_length(self):
        return self.client.server_info()[self.table].max_size

    def sample_by_numbers(self, n):
        data = self.client.sample(self.table, n)
        # Each sample consists of only one timestep
        samples = [x[0].data for x in data]

        # Extracts the data from the sample
        train_inputs = []
        for sample in samples:
            train_inputs.append(
                utils.TrainInput(environment=sample[0], state=sample[1], value=sample[2], policy=sample[3]))
        return train_inputs

    def dataset(self):
        return reverb.TimestepDataset(self.client, self.table)

    def length(self):
        return self.client.server_info()[self.table].current_size

    def sample_by_fraction(self, p):
        return self.sample_by_numbers(math.ceil(p * self.length()))

    @property
    def supports_dataset(self):
        return False


class Broadcaster(abc.ABC):
    @abc.abstractmethod
    def broadcast(self, path):
        pass


class ProcessesBroadcaster(Broadcaster):

    def __init__(self, processes: List[spawn.Process]):
        self.processes = processes

    def broadcast(self, path):
        for process in self.processes:
            process.queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_MESSAGE, path))
        pass

    def request_exit(self):
        for process in self.processes:
            process.queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_CLOSE, None))

        for process in self.processes:
            while not process.queue.empty():
                process.queue.get_nowait()
            process.join()
        pass


class ModelReceiver(abc.ABC):
    @abc.abstractmethod
    def receive(self):
        pass
