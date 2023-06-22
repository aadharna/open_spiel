import abc
import json
import socket
import threading
import time
from typing import List, Dict, TypedDict, Callable

import numpy as np
import reverb
from open_spiel.python.utils import spawn, stats
from .. import utils
from . import queue as queue_lib


class TrajectoryCollector(abc.ABC):

    @abc.abstractmethod
    def collect(self):
        pass

    @property
    def metadata(self) -> Dict:
        return {}


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
                 wait_duration: float = 0.01, block_until_ready: bool = False):
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
        self.block_until_ready = block_until_ready
        self._stats = [{} for _ in range(len(processes))]
        pass

    def combiner(self):
        while True:
            found = 0
            for thread_id, actor_process in enumerate(self.processes):
                try:
                    if self.block_until_ready:
                        yield (thread_id, actor_process.queue.get())
                    else:
                        yield (thread_id, actor_process.queue.get_nowait())
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(self.wait_duration)  # 10ms
        pass

    @property
    def stats(self) -> List[Dict]:
        return self._stats

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
        for thread_id, queue_content in self.combiner():
            message_type, message = queue_content
            if message_type == queue_lib.MessageTypes.QUEUE_ANALYSIS:
                self._stats[thread_id] = message
                continue

            # Ignore the queue close message. It is used as a confirmation that the thread is done.
            elif message_type == queue_lib.MessageTypes.QUEUE_CLOSE:
                continue

            trajectory = message
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
                                 value=p1_outcome, policy=s.policy) for s in trajectory.states)

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return train_inputs, self.CollectedMetadata(num_trajectories=num_trajectories,
                                                    num_states=num_states,
                                                    outcomes=outcomes.data,
                                                    game_lengths=game_lengths.as_dict,
                                                    game_lengths_hist=game_lengths_hist.data,
                                                    value_accuracies=[v.as_dict for v in value_accuracies],
                                                    value_predictions=[v.as_dict for v in value_predictions])


class ProcessOrchestrator:
    def __init__(self, processes: List[spawn.Process]):
        self.processes = processes
        pass

    def broadcast(self, message):
        for process in self.processes:
            process.queue.put(message)
        pass

    def collect(self):
        return [process.queue.get() for process in self.processes]


class PeriodicThread(threading.Thread):
    def __init__(self, period: float, func: Callable, *args, **kwargs):
        super().__init__()
        self.period = period
        self.func = func
        self._stop_event = threading.Event()
        self.args = args
        self.kwargs = kwargs
        pass

    def run(self):
        while not self._stop_event.is_set():
            self.func(*self.args, **self.kwargs)
            self._stop_event.wait(self.period)
        pass

    def stop(self):
        self._stop_event.set()
        pass


class ActorsGrpcOrchestrator(ProcessOrchestrator):

    def __init__(self, actors: List[spawn.Process], server_address: str, server_port: int, table: str,*,
                 server_max_workers: int,
                 max_buffer_size: int, max_game_length, request_length=1024, server_timeout: int = 30,
                 max_collection_time: int = None, stats_frequency: int = 60, stats_file: str = None,
                 collection_period: float = None):
        super().__init__(actors)
        self.collector = ProcessesTrajectoryCollector(actors, max_game_length=max_game_length,
                                                      learn_rate=request_length, block_until_ready=False)
        self.server_address = server_address
        self.server_port = server_port
        self.table = table
        self.server_max_workers = server_max_workers
        self.server_timeout = server_timeout
        self.client = reverb.Client(f"{self.server_address}:{self.server_port}")
        self.max_buffer_size = max_buffer_size
        self._stop_request = False
        self.max_collection_time = max_collection_time
        self.stats_frequency = stats_frequency
        if stats_file is None:
            stats_file = "actor"
        self.stats_file = f"{stats_file}_{socket.gethostname()}.json"
        self.stats_save_thread = PeriodicThread(self.stats_frequency, self.save_stats)
        self.stats_save_thread.start()
        if collection_period is not None:
            self.collection_thread = PeriodicThread(collection_period, self.collect)
            self.collection_thread.start()
        pass

    def _time_to_stop(self, start_time):
        if self.max_collection_time is None:
            return False
        return time.time() - start_time > self.max_collection_time

    def save_stats(self):
        with open(self.stats_file, "a") as file:
            json.dump(self.collector.stats, file)
            # Add a comma to separate the json objects
            file.write(",\n")

    def collect(self):
        start_time = None
        if self.max_collection_time is not None:
            start_time = time.time()
        while not self._stop_request and not self._time_to_stop(start_time):
            train_inputs, metadata = self.collector.collect()
            batch_size=len(train_inputs)
            #This guarantees that the inputs does not exceed the max buffer size
            for index in range((batch_size + self.max_buffer_size - 1) // self.max_buffer_size):
                with self.client.writer(self.max_buffer_size) as writer:
                    for train_input in train_inputs[index * self.max_buffer_size:(index + 1) * self.max_buffer_size]:
                        # The order of the keys is important
                        # It appears that the ordering of reverb is not consistent for dictionaries
                        # This is why we use a list
                        train_input_dict = [
                            np.array(train_input.environment, dtype=np.float32),
                            np.array(train_input.state, dtype=np.float32),
                            np.array(train_input.value, dtype=np.float32),
                            np.array(train_input.policy, dtype=np.float32)
                        ]
                        writer.append(train_input_dict)
                        writer.create_item(self.table, num_timesteps=1, priority=1.0)
                    writer.flush()
        pass

    def stop(self):
        self._stop_request = True
        pass


class EvaluatorOrchestrator(ProcessOrchestrator):
    def __init__(self, evaluators: List[spawn.Process], max_game_length: int, max_collection_time: int = None,
                 stats_frequency: int = 60, stats_file: str = None):
        super().__init__(evaluators)
        self.collector = ProcessesTrajectoryCollector(evaluators, max_game_length=max_game_length, learn_rate=1,
                                                      block_until_ready=False)
        self._stop_request = False
        self.max_collection_time = max_collection_time
        self.stats_frequency = stats_frequency
        if stats_file is None:
            stats_file = "evaluator"
        self.stats_file = f"{stats_file}_{socket.gethostname()}.json"

    def _time_to_stop(self, start_time):
        if self.max_collection_time is None:
            return False
        return time.time() - start_time > self.max_collection_time

    @property
    def metadata(self):
        return 0

    def collect(self):
        start_time = None
        if self.max_collection_time is not None:
            start_time = time.time()
            _, metadata = self.collector.collect()
            time.sleep(self.stats_frequency)

        pass

    def stop(self):
        self._stop_request = True
        pass
