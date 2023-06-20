import abc
import math
import time
from typing import List, Dict, TypedDict, Union

from open_spiel.python.utils import spawn, stats
from . import utils, model as model_lib
import random
from .services import actor


class TrajectoryCollector(abc.ABC):

    @abc.abstractmethod
    def collect(self, replay_buffer: utils.Buffer):
        pass


class DefaultTrajectoryCollector(TrajectoryCollector):
    def collect(self, replay_buffer):
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
                 wait_duration: float = 0.01):
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
        pass

    def combiner(self):
        while True:
            found = 0
            for actor_process in self.processes:
                try:
                    yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(self.wait_duration)  # 10ms
        pass

    def collect(self, replay_buffer):
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

            replay_buffer.extend(
                model_lib.TrainInput(s.environment, s.state, s.policy, p1_outcome)
                for s in trajectory.states)

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return self.CollectedMetadata(num_trajectories=num_trajectories,
                                      num_states=num_states,
                                      outcomes=outcomes.data,
                                      game_lengths=game_lengths.as_dict,
                                      game_lengths_hist=game_lengths_hist.data,
                                      value_accuracies=[v.as_dict for v in value_accuracies],
                                      value_predictions=[v.as_dict for v in value_predictions])


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
        self.current_analysis_data = self.trajectory_collector.collect(self.replay_buffer)

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


class gRPCReplayBuffer(ReplayBufferAdapter):
    pass


class ModelBroadcaster(abc.ABC):
    @abc.abstractmethod
    def broadcast(self, path):
        pass


class QueuesModelBroadcaster(ModelBroadcaster):

    def __init__(self, processes: List[spawn.Process]):
        self.processes = processes

    def broadcast(self, path):
        for process in self.processes:
            process.queue.put(path)
        pass


class ModelReceiver(abc.ABC):
    @abc.abstractmethod
    def receive(self):
        pass
