import abc
import argparse
import collections
import datetime
import functools
import traceback
from typing import Iterable, List, Union, Tuple

import numpy as np
import reverb
from open_spiel.python.utils import file_logger, spawn
import open_spiel.python.algorithms.mcts as mcts
import random
import tensorflow as tf

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


def get_reverb_selector(name):
    if name == "random" or name == "uniform":
        return reverb.selectors.Uniform
    elif name == "fifo":
        return reverb.selectors.Fifo
    elif name == "priority":
        return reverb.selectors.Prioritized
    elif name == "lifo":
        return reverb.selectors.Lifo
    else:
        raise ValueError("Unknown selector: {}".format(name))


def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))



def recursive_namespace_todict(ns):
    """Converts a namespace object to a dictionary, recursively."""
    d = {}
    if isinstance(ns, argparse.Namespace):
        ns = vars(ns)

    if isinstance(ns, dict):
        return {k: recursive_namespace_todict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [recursive_namespace_todict(v) for v in ns]
    else:
        return ns


def reduce_lists(ns):
    result = {}
    if isinstance(ns, dict):
        for k, v in ns.items():
            r = reduce_lists(v)
            result[k] = r
        return result
    if isinstance(ns, list):
        mapper = map(str, ns)
        return " ".join(mapper)
    return ns


def game_complete_name(game_config) -> Union[Tuple[str], Tuple[str, str]]:
    game_dict = reduce_lists(recursive_namespace_todict(game_config))
    if "generator" in game_dict:
        game_dict["generator_params"] = game_dict["generator"]["params"]
        game_dict["generator"] = game_config.generator.name
    game_params = []
    game_name = game_dict["name"]
    game_dict.pop("name")
    game_dict.pop("fix_environment")
    if len(game_dict) == 0:
        return (game_name,)
    else:
        return (game_name, game_dict)


def expand_arguments(fn, *args, **kwargs):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, argparse.Namespace) or isinstance(args, dict):
        args = vars(args)
        return fn(**args, **kwargs)
    elif isinstance(args, Iterable):
        return fn(*args, **kwargs)
    else:
        return fn(args, **kwargs)


def get_reverb_signature():
    return [tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
                       tf.TensorSpec(shape=(1,), dtype=tf.float32),
                       tf.TensorSpec(shape=(), dtype=tf.float32),
                       tf.TensorSpec(shape=(None,), dtype=tf.float32)]

class Watched(abc.ABC):
    def __init__(self, config, num=None, name=None, *, to_stdout=False,log_directory=None):
        self.config = config
        self.num = num
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.to_stdout = to_stdout
        self.path = log_directory
        pass

    def start(self, *args, **kwargs):
        name = self.name
        if self.num is not None:
            name += "-" + str(self.num)
        if self.path is None:
            path=self.config.path
        else:
            path=self.path
        with file_logger.FileLogger(path, name, self.config.quiet) as logger:
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

    def __init__(self,graph_size:int=None,edges_count:int=None):
        self.states = []
        self.returns = None
        self.graph_size=graph_size
        self.edges_count=edges_count

    def add(self, information_state, action, policy):
        self.states.append((information_state, action, policy))


def is_mcts_bot(bot):
    """Check if a bot is an MCTS bot."""
    return isinstance(bot, mcts.MCTSBot)


def _game_result_from_perspective(reward,player):
    if player==0:
        return reward
    else:
        return -reward

def play_game(logger, game_num, game, bots, temperature, temperature_drop, fix_environment=False):
    """Play one game, return the trajectory."""
    actions = []
    if fix_environment:
        state = game.new_initial_state()
    else:
        state = game.new_initial_environment_state()
    trajectory = Trajectory(edges_count=state.count_edges(),graph_size=state.graph_size())
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
                policy = np.zeros(state.graph_size())
                for c in root.children:
                    policy[c.action] = c.explore_count
                policy = policy ** (1 / temperature)
                policy /= policy.sum()
                if len(actions) >= temperature_drop:
                    action = root.best_child().action
                else:
                    action = np.random.choice(len(policy), p=policy)
                trajectory.states.append(
                    TrajectoryState(*nested_reshape(state.observation_tensor(), state.observation_tensor_shapes_list()),
                                    state.current_player(), action, policy,
                                    value=root.total_reward / root.explore_count))
            else:
                action = bots[state.current_player()].step(state)
                policy = np.zeros(state.graph_size())
                policy[action] = 1
                trajectory.states.append(
                    TrajectoryState(*nested_reshape(state.observation_tensor(), state.observation_tensor_shapes_list()),
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



def nested_dict_to_namespace(nested_dict):
    """Converts a nested dict to a namespace."""
    if isinstance(nested_dict,argparse.Namespace):
        return nested_dict_to_namespace(vars(nested_dict))
    if type(nested_dict) != dict and type(nested_dict) != list:
        return nested_dict
    namespace = argparse.Namespace()
    if type(nested_dict) is dict:
        for key, value in nested_dict.items():
            if isinstance(key,str):
                setattr(namespace, key, nested_dict_to_namespace(value))

    elif type(nested_dict) is list:
        namespace = [None]*len(nested_dict)
        for i, value in enumerate(nested_dict):
            namespace[i]=nested_dict_to_namespace(value)
    return namespace


def compatibility_mode(config):
    config.fix_environment = config.game.fix_environment
    config.temperature = config.mcts.temperature
    config.temperature_drop = config.mcts.temperature_drop
    config.mcts.policy_epsilon = config.mcts.policy_epsilon
    config.mcts.policy_alpha = config.mcts.policy_alpha
    config.mcts.max_simulations = config.mcts.max_simulations
    config.mcts.temperature_drop = config.mcts.temperature_drop
    if "max_moves" in vars(config.game):
        config.max_moves = config.game.max_moves
    config.nn_model = config.model.architecture
    config.nn_width = config.model.width
    config.nn_depth = config.model.depth
    config.training_batch_size = config.training.batch_size
    config.learning_rate = config.training.learning_rate
    config.weight_decay = config.training.weight_decay
    config.checkpoint_freq = config.training.checkpoint_freq
    config.max_steps = config.training.max_steps
    config.steps_per_epoch = config.training.steps_per_epoch
    config.epochs_per_iteration = config.training.epochs_per_iteration
    config.evaluation_window = config.services.evaluators.evaluation_window
    config.regularization = config.training.weight_decay
    config.policy_epsilon = config.mcts.policy_epsilon
    config.policy_alpha = config.mcts.policy_alpha
    config.uct_c = config.mcts.uct_c
    config.replay_buffer_size = config.replay_buffer.buffer_size
    config.replay_buffer_reuse = config.replay_buffer.reuse
    config.max_simulations = config.mcts.max_simulations
    config.train_batch_size = config.training.batch_size




INPUT_NAMES=("environment","state")
OUTPUT_NAMES=("value","policy")
TRAIN_NAMES=("environment","state","value","policy")


def get_winner_name(mean_payoff) -> str:
    if mean_payoff > 0:
        return "Max"
    elif mean_payoff < 0:
        return "Min"
    else:
        return "Draw"