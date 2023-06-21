import abc
import collections
import functools
import traceback
from typing import Iterable, List

import numpy as np
from open_spiel.python.utils import file_logger, spawn
import open_spiel.python.algorithms.mcts as mcts
import random

def watcher(fn):
    """A decorator to fn/processes that gives a logger and logs exceptions."""

    @functools.wraps(fn)
    def _watcher(*, config, num=None, **kwargs):
        """Wrap the decorated function."""
        name = fn.__name__
        if num is not None:
            name += "-" + str(num)
        with file_logger.FileLogger(config.path, name, config.quiet) as logger:
            print("{} started".format(name))
            logger.print("{} started".format(name))
            try:
                return fn(config=config, logger=logger, **kwargs)
            except Exception as e:
                logger.print("\n".join([
                    "",
                    " Exception caught ".center(60, "="),
                    traceback.format_exc(),
                    "=" * 60,
                ]))
                print("Exception caught in {}: {}".format(name, e))
                raise
            finally:
                logger.print("{} exiting".format(name))
                print("{} exiting".format(name))

    return _watcher


class Watched(abc.ABC):
    def __init__(self, config, num=None, name=None, *, to_stdout=False):
        self.config = config
        self.num = num
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.to_stdout = to_stdout
        pass

    def start(self, *args, **kwargs):
        name = self.name
        if self.num is not None:
            name += "-" + str(self.num)
        with file_logger.FileLogger(self.config.path, name, self.config.quiet) as logger:
            logger.also_to_stdout = self.to_stdout
            print("{} started".format(name))
            logger.print("{} started".format(name))
            try:
                return self.run(*args,logger=logger, **kwargs)
            except Exception as e:
                logger.print("\n".join([
                    "",
                    " Exception caught ".center(60, "="),
                    traceback.format_exc(),
                    "=" * 60,
                ]))
                print("Exception caught in {}: {}".format(name, e))
                raise
            finally:
                logger.print("{} exiting".format(name))
                print("{} exiting".format(name))
        pass

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass



# TODO: Add mean payoffs to the trajectory
class TrajectoryState(object):
    """A particular point along a trajectory."""

    def __init__(self, environment, state, current_player, action, policy,
                 value):
        self.environment = environment
        self.state = state
        self.current_player = current_player
        self.action = action
        self.policy = policy
        self.value = value


class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    self.data[:-self.max_size] = []

  def sample(self, count):
    return random.sample(self.data, count)



class Config(collections.namedtuple(
    "Config", [
        "game",
        "path",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "replay_buffer_size",
        "replay_buffer_reuse",
        "max_steps",
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",
        "uct_c",
        "max_simulations",
        "policy_alpha",
        "policy_epsilon",
        "temperature",
        "temperature_drop",
        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",
        "verbose",
        "quiet",
        "fix_environment",
        "version",
        "grpc",
        "grpc_address",
        "grpc_port",
        "grpc_table",
        "grpc_min_size",
        "steps_per_epoch",
        "epochs_per_iteration",
    ])):
  """A config for the model/experiment."""
  @property
  def architecture(self):
      return self.nn_model
  pass
  @property
  def regularization(self):
    return self.weight_decay

class Trajectory(object):
    """A sequence of observations, actions and policies, and the outcomes."""

    def __init__(self):
        self.states = []
        self.returns = None

    def add(self, information_state, action, policy):
        self.states.append((information_state, action, policy))


def is_mcts_bot(bot):
    """Check if a bot is an MCTS bot."""
    return isinstance(bot, mcts.MCTSBot)


def play_game(logger, game_num, game, bots, temperature, temperature_drop, fix_environment=False):
    """Play one game, return the trajectory."""
    trajectory = Trajectory()
    actions = []
    if fix_environment:
        state = game.new_initial_state()
    else:
        state = game.new_initial_environment_state()
    random_state = np.random.RandomState()
    logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
    logger.opt_print("Initial state:\n{}".format(state))
    while not state.is_terminal():
        if state.is_chance_node():
            # For chance nodes, rollout according to chance node's probability
            # distribution
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random_state.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            if is_mcts_bot(bots[state.current_player()]):
                root = bots[state.current_player()].mcts_search(state)
                policy = np.zeros(game.num_distinct_actions())
                for c in root.children:
                    policy[c.action] = c.explore_count
                policy = policy ** (1 / temperature)
                policy /= policy.sum()
                if len(actions) >= temperature_drop:
                    action = root.best_child().action
                else:
                    action = np.random.choice(len(policy), p=policy)
                trajectory.states.append(
                    TrajectoryState(*nested_reshape(state.observation_tensor(), game.observation_tensor_shapes_list()),
                                    state.current_player(),
                                    action, policy, root.total_reward / root.explore_count))
            else:
                action = bots[state.current_player()].step(state)
                policy = np.zeros(game.num_distinct_actions())
                policy[action] = 1
                trajectory.states.append(
                    TrajectoryState(*nested_reshape(state.observation_tensor(), game.observation_tensor_shapes_list()),
                                    state.current_player(),
                                    action, policy, value=None))
            action_str = state.action_to_string(state.current_player(), action)
            actions.append(action_str)
            logger.opt_print("Player {} sampled action: {}".format(state.current_player(), action_str))
            state.apply_action(action)
    logger.opt_print("Next state:\n{}".format(state))

    trajectory.returns = state.returns()
    logger.print("Game {}: Returns: {}; Actions: {}".format(
        game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
    return trajectory


class AlmostUniversalHasher:

    def __init__(self,x,mod):
        self.x = x
        self.mod = mod

    def hash(self,data):
        if isinstance(data,Iterable):
            return sum([self.hash(t)* self.x.__pow__(i,self.mod) for i,t in enumerate(data)]) % self.mod
        else:
            return hash(data) % self.mod

    @classmethod
    def deterministic_instance(cls):
        return cls(x=2654435761,mod=18446744073709551557)
def nested_reshape(flat_array, shapes_list):
    arrays=[]
    start=0
    for shape in shapes_list:
        size=np.prod(shape)
        arrays.append(np.reshape(flat_array[start:start+size],shape))
        start+=size
    if start!=len(flat_array):
        raise ValueError("Shapes don't match")
    return arrays

class TrainInput(collections.namedtuple(
    "TrainInput", ["environment", "state", "value", "policy"])):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs):
        environment= np.stack([t.environment for t in train_inputs])
        state= np.stack([t.state for t in train_inputs])
        value= np.stack([t.value for t in train_inputs])
        policy= np.stack([t.policy for t in train_inputs])
        return TrainInput(
            environment=np.array(environment, dtype=np.float32),
            state=np.array(state, dtype=np.int32),
            value=np.expand_dims(value, 1),
            policy=np.array(policy),
        )

INPUT_NAMES=("environment","state")
OUTPUT_NAMES=("value","policy")
TRAIN_NAMES=("environment","state","value","policy")