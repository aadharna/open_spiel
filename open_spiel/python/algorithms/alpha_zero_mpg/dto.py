import abc
import time
from typing import List

import open_spiel.python.utils.spawn as spawn
from . import utils
from .services import actor


class TrajectoryCollector(abc.ABC):

    @abc.abstractmethod
    def collect(self):
        pass


class DefaultTrajectoryCollector(TrajectoryCollector):
    def collect(self):
        pass


class ProcessesTrajectoryCollector(TrajectoryCollector):
    """
    Collect trajectory from processes
    """

    def __init__(self, processes):
        self.processes = processes
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
                time.sleep(0.01)  # 10ms
        pass

    def collect(self):
        num_trajectories = 0
        num_states = 0
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
        return num_trajectories, num_states


class ReplayBufferAdapter(abc.ABC):
    def __init__(self):
        pass

    def sample(self,n):
        if type(n) == float:
            if n < 0:
                raise ValueError(f"The fraction cannot be negative")
            if n <= 1:
                return self.sample_by_fraction(n)
        return self.sample_by_numbers(n)


    @abc.abstractmethod
    def sample_by_fraction(self,p):
        pass

    @abc.abstractmethod
    def sample_by_numbers(self,n):
        pass


    @abc.abstractmethod
    def length(self):
        pass


    @abc.abstractmethod
    def length(self):
        pass
    def __len__(self):
        return self.length()

    def __bool__(self):
        return self.length() == 0


class QueuesReplayBuffer(ReplayBufferAdapter):

    def __init__(self,replay_buffer_size:int,replay_buffer_reuse:int,actors:List[actor.Actor]):
        super().__init__()
        self.replay_buffer_size=replay_buffer_size
        self.replay_buffer_reuse=replay_buffer_reuse
        self.replay_buffer=utils.Buffer(replay_buffer_size)
        self.trajectory_collector=ProcessesTrajectoryCollector(actors)

class gRPCReplayBuffer(ReplayBufferAdapter):
    pass


class ModelSender(abc.ABC):
    @abc.abstractmethod
    def send(self):
        pass


class ModelReceiver(abc.ABC):
    @abc.abstractmethod
    def receive(self):
        pass
