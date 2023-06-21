import abc
import math
import time
from typing import List, Dict, TypedDict, Union

import numpy as np
from open_spiel.python.utils import spawn, stats

import open_spiel.python.algorithms.alpha_zero_mpg.utils
from . import utils, model as model_lib
import random
import reverb
from .services import actor


class TrajectoryCollector(abc.ABC):

    @abc.abstractmethod
    def collect(self):
        pass



class ProcessesTrajectoryCollector(TrajectoryCollector):
    """
    Collect trajectory from processes
    """
    STAGE_COUNT_DEFAULT: int = 7

    class CollectedMetadata(TypedDict):
        num_trajectories: int
        num_states: int
        outcomes: stats.HistogramNamed
        game_lengths: stats.BasicStats
        game_lengths_hist: stats.HistogramNumbered
        value_accuracies: List[stats.BasicStats]
        value_predictions: List[stats.BasicStats]
        pass

    def __init__(self, processes: List[spawn.Process], stage_count: int = None, max_game_length=None, learn_rate=None,
                 wait_duration: float = 0.01,block_until_ready:bool=False):
        self.processes = processes
        if stage_count is None:
            stage_count = self.STAGE_COUNT_DEFAULT
        self.stage_count = stage_count
        if max_game_length is None:
            raise ValueError(f"max_game_length should be given")
        if learn_rate is None:
            raise ValueError(f"learn_rate should be given")
        self.max_game_length = max_game_length
        self.learn_rate = learn_rate
        self.wait_duration = wait_duration
        self.block_until_ready=block_until_ready
        pass

    def combiner(self):
        while True:
            found = 0
            for actor_process in self.processes:
                try:
                    if self.block_until_ready:
                        yield actor_process.queue.get()
                    else:
                        yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(self.wait_duration)  # 10ms
        pass

    def collect(self):
        num_trajectories = 0
        num_states = 0
        game_lengths = stats.BasicStats()
        stage_count = self.stage_count
        max_game_length = self.max_game_length
        learn_rate = self.learn_rate
        game_lengths_hist = stats.HistogramNumbered(max_game_length + 1)
        outcomes = stats.HistogramNamed(["PlayerMax", "PlayerMin", "Draw"])
        value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
        value_predictions = [stats.BasicStats() for _ in range(stage_count)]
        train_inputs = []
        for trajectory in self.combiner():
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            train_inputs.extend(
                utils.TrainInput(environment=s.environment, state=s.state,
                                 value=p1_outcome,policy=s.policy) for s in trajectory.states)

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return train_inputs,self.CollectedMetadata(num_trajectories=num_trajectories,
                                      num_states=num_states,
                                      outcomes=outcomes.data,
                                      game_lengths=game_lengths.as_dict,
                                      game_lengths_hist=game_lengths_hist.data,
                                      value_accuracies=[v.as_dict for v in value_accuracies],
                                      value_predictions=[v.as_dict for v in value_predictions])


class ProcessOrchestrator:
    def __init__(self,processes:List[spawn.Process]):
        self.processes=processes
        pass

    def broadcast(self,message):
        for process in self.processes:
            process.queue.put(message)
        pass

    def collect(self):
        return [process.queue.get() for process in self.processes]


class ActorsGrpcOrchestrator(ProcessOrchestrator):

    def __init__(self,actors:List[spawn.Process],server_address:str,server_port:int,table:str,server_max_workers:int,
                 max_buffer_size:int,max_game_length,request_length=1024,server_timeout:int=30):
        super().__init__(actors)
        self.collector=ProcessesTrajectoryCollector(actors,max_game_length=max_game_length,learn_rate=request_length,block_until_ready=False)
        self.server_address=server_address
        self.server_port=server_port
        self.table=table
        self.server_max_workers=server_max_workers
        self.server_timeout=server_timeout
        self.client=reverb.Client(f"{self.server_address}:{self.server_port}")
        self.max_buffer_size=max_buffer_size
        self._stop_request=False

    def collect(self):
        while not self._stop_request:
            train_inputs,metadata=self.collector.collect()
            with self.client.writer(self.max_buffer_size) as writer:
                for train_input in train_inputs:
                    #The order of the keys is important
                    train_input_dict=[
                        np.array(train_input.environment,dtype=np.float32),
                        np.array(train_input.state,dtype=np.float32),
                        np.array(train_input.value,dtype=np.float32),
                        np.array(train_input.policy,dtype=np.float32)
                    ]
                    writer.append(train_input_dict)
                    writer.create_item(self.table, num_timesteps=1,priority=1.0)
                writer.flush()
        pass

    def stop(self):
        self._stop_request=True
        pass

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
        train_data,self.current_analysis_data = self.trajectory_collector.collect()
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
    def __init__(self,address:str,port:int,table:str):
        super().__init__()
        self.address=address
        self.port=port
        self.table=table
        self.client=reverb.Client(f"{self.address}:{self.port}")
    pass

    def update(self):
        pass

    @property
    def learn_rate(self):
        raise NotImplementedError("The learn rate is not implemented for this replay buffer")

    def max_length(self):
        return self.client.server_info()[self.table].max_size


    def sample_by_numbers(self, n):
        data= self.client.sample(self.table,n)
        #Each sample consists of only one timestep
        samples= [x[0].data for x in data]

        #Extracts the data from the sample
        train_inputs=[]
        for sample in samples:
            train_inputs.append(utils.TrainInput(environment=sample[0],state=sample[1],value=sample[2],policy=sample[3]))
        return train_inputs

    def dataset(self):
        return reverb.TimestepDataset(self.client,self.table)


    def length(self):
        return self.client.server_info()[self.table].current_size

    def sample_by_fraction(self, p):
        return self.sample_by_numbers(math.ceil(p*self.length()))

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
            process.queue.put(path)
        pass

    def request_exit(self):
        for process in self.processes:
            process.queue.put("")
        pass


class ModelReceiver(abc.ABC):
    @abc.abstractmethod
    def receive(self):
        pass
