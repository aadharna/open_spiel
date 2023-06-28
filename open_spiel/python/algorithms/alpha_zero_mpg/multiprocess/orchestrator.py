import abc
import json
import socket
import threading
import time
from typing import List, Dict, TypedDict, Callable, Any

import numpy as np
import reverb
from open_spiel.python.utils import spawn, stats
from .. import utils
from . import queue as queue_lib
import os


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
                 wait_duration: float = 0.01, block_until_ready: bool = False,sampler="trajectory",
                 value_target=None):
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
        self.on_heartbeat: List[Callable[[int, Any], None]] = []
        self.on_close: List[Callable[[int, Any], None]] = []
        self.on_analysis: List[Callable[[int, Any], None]] = []
        self.sampler=sampler
        self.value_target=value_target
        if self.value_target not in ["winner","mean_payoff","value"]:
            raise ValueError(f"value_target should be either winner or mean_payoff. Got {self.value_target}")
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

    def add_on_heartbeat(self, callback: Callable[[int, Any], None]):
        self.on_heartbeat.append(callback)
        pass

    def add_on_close(self, callback: Callable[[int, Any], None]):
        self.on_close.append(callback)
        pass

    def add_on_analysis(self, callback: Callable[[int, Any], None]):
        self.on_analysis.append(callback)

    def sample_train_inputs(self,trajectory):
        mask=np.zeros(len(trajectory.states),dtype=np.int32)
        if self.sampler=="trajectory":
            mask[:]=1
        elif self.sampler=="last":
            mask[-1]=1
        elif self.sampler=="first":
            mask[0]=1
        elif self.sampler=="random":
            mask_indexes=np.random.choice(len(trajectory.states),1,replace=False)
            mask[mask_indexes]=1
        elif isinstance(self.sampler,int):
            mask_indexes=np.random.choice(len(trajectory.states),min(self.sampler,len(trajectory.states)),replace=False)
            mask[mask_indexes]=1
        elif isinstance(self.sampler,float):
            indexes_count=np.random.binomial(n=len(trajectory.states),p=self.sampler)
            mask_indexes=np.random.choice(len(trajectory.states),indexes_count,replace=False)
            mask[mask_indexes]=1
        else:
            raise ValueError(f"Unknown sampler {self.sampler}")
        states:List[utils.TrainInput]=[]
        for index,s in enumerate(trajectory.states):
            if mask[index]:
                if self.value_target == "winner":
                    valuation=trajectory.returns[0]
                elif self.value_target== "mean_payoff":
                    valuation=trajectory.mean_payoff
                else:
                    valuation=trajectory.values[index]
                states.append(utils.TrainInput(environment=s.environment, state=s.state, value=valuation, policy=s.policy))
        return states


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
            print(message_type)
            if message_type == queue_lib.MessageTypes.QUEUE_ANALYSIS:
                self._stats[thread_id] = message
                for callback in self.on_analysis:
                    callback(thread_id, message)
                continue

            # Ignore the queue close message. It is used as a confirmation that the thread is done.
            elif message_type == queue_lib.MessageTypes.QUEUE_CLOSE:
                for callback in self.on_close:
                    callback(thread_id, message)
                continue
            elif message_type == queue_lib.MessageTypes.QUEUE_HEARTBEAT:
                print(f"Received heartbeat from {thread_id}")
                for callback in self.on_heartbeat:
                    callback(thread_id, message)
                continue

            #if self.sampler=="trajectory":

            trajectory = message


            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            samples=self.sample_train_inputs(trajectory)
            num_samples=len(samples)
            train_inputs.extend(samples)
            num_trajectories += 1
            num_states += num_samples
            game_lengths.add(num_samples)
            game_lengths_hist.add(num_samples)


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

    def heartbeat(self):
        for process in self.processes:
            process.queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_HEARTBEAT, None))
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

    def __init__(self, actors: List[spawn.Process], config, max_game_length: int,*,working_directory: str = None):
        super().__init__(actors)
        server_address = config.replay_buffer.implementation.address
        server_port = config.replay_buffer.implementation.port
        table = config.replay_buffer.implementation.table
        server_max_workers = 10
        server_timeout = 10
        max_buffer_size = config.replay_buffer.buffer_size
        request_length = config.services.actors.request_length
        max_collection_time = config.services.actors.max_collection_time
        stats_frequency = config.services.actors.stats_frequency

        collection_period = config.services.actors.collection_period
        self.collector = ProcessesTrajectoryCollector(actors, max_game_length=max_game_length,
                                                      learn_rate=request_length, block_until_ready=False,
                                                      sampler=config.replay_buffer.writer_sampler,value_target=config.replay_buffer.value_target)
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
        self.config=config
        stats_file = config.services.evaluators.stats_file
        if stats_file is None:
            stats_file = "actor-stats.jsonl"

        self.working_directory = working_directory or config.path
        self.stats_file = os.path.join(self.working_directory, stats_file)

        self.stats_save_thread = PeriodicThread(self.stats_frequency, self.save_stats)
        self.stats_save_thread.start()
        self.request_length = request_length
        self.collection_period = collection_period
        if collection_period is not None:
            self.collection_thread = PeriodicThread(collection_period, self.collect)
            self.collection_thread.start()
        else:
            self.collection_thread = None

    def _time_to_stop(self, start_time):
        if self.max_collection_time is None:
            return False
        return time.time() - start_time > self.max_collection_time

    def save_stats(self):
        with open(self.stats_file, "a") as file:
            json.dump(self.collector.stats, file, default=utils.json_serializer)
            # Add a comma to separate the json objects
            file.write("\n")

    def collect(self):
        start_time = None
        if self.max_collection_time is not None:
            start_time = time.time()
        while not self._stop_request and not self._time_to_stop(start_time):
            train_inputs, metadata = self.collector.collect()
            batch_size = len(train_inputs)
            # This guarantees that the inputs does not exceed the max buffer size
            for index in range((batch_size + self.request_length - 1) // self.request_length):
                with self.client.writer(max_sequence_length=1) as writer:
                    for train_input in train_inputs[index * self.request_length:(index + 1) * self.request_length]:
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
        self.broadcast(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_CLOSE, None))
        self._stop_request = True
        if self.collection_thread is not None:
            self.collection_thread.stop()
        self.stats_save_thread.stop()
        pass

    def add_on_heartbeat(self, func: Callable[[int, Any], None]):
        self.collector.add_on_heartbeat(func)

    def add_on_analysis(self, func: Callable[[int, Any], None]):
        self.collector.add_on_analysis(func)

    def add_on_close(self, func: Callable[[int, Any], None]):
        self.collector.add_on_close(func)


class EvaluatorOrchestrator(ProcessOrchestrator):
    def __init__(self, evaluators: List[spawn.Process], config, max_game_length: int,*,working_directory=None):
        super().__init__(evaluators)
        max_collection_time = config.services.evaluators.max_collection_time
        stats_frequency = config.services.evaluators.stats_frequency
        stats_file = config.services.evaluators.stats_file

        self.collector = ProcessesTrajectoryCollector(evaluators, max_game_length=max_game_length,
                                                      learn_rate=1, block_until_ready=False,
                                                      sampler=config.replay_buffer.writer_sampler,
                                                      value_target=config.replay_buffer.value_target)
        self.config = config
        self._stop_request = False
        self.max_collection_time = max_collection_time
        self.stats_frequency = stats_frequency
        if stats_file is None:
            stats_file = "evaluator"
        self.working_directory = working_directory or config.path
        self.stats_file = os.path.join(self.working_directory, stats_file)
        self.stats_save_thread = PeriodicThread(self.stats_frequency, self.save_stats)
        self.stats_save_thread.start()

    def _time_to_stop(self, start_time):
        if self.max_collection_time is None:
            return False
        return time.time() - start_time > self.max_collection_time

    def save_stats(self):
        with open(self.stats_file, "a") as file:
            json.dump(self.collector.stats, file, default=utils.json_serializer)
            # Add a comma to separate the json objects
            file.write(",\n")

    def collect(self):
        start_time = None
        while not self._stop_request and not self._time_to_stop(start_time):
            start_time = time.time()
            _, metadata = self.collector.collect()
            time.sleep(self.stats_frequency)

        pass

    def stop(self):
        self.broadcast(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_CLOSE, None))
        self._stop_request = True
        pass

    def add_on_heartbeat(self, func: Callable[[int, Any], None]):
        self.collector.add_on_heartbeat(func)

    def add_on_analysis(self, func: Callable[[int, Any], None]):
        self.collector.add_on_analysis(func)

    def add_on_close(self, func: Callable[[int, Any], None]):
        self.collector.add_on_close(func)
