import abc
import functools
import traceback

import numpy as np
from open_spiel.python.utils import file_logger
from open_spiel.python.algorithms.alpha_zero_mpg.model import nested_reshape
import open_spiel.python.algorithms.mcts as mcts


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
    def __init__(self, config, num=None, name=None):
        self.config = config
        self.num = num
        if name is None:
            name = self.__class__.__name__
        self.name = name
        pass

    def start(self, *args, **kwargs):
        name = self.name
        if self.num is not None:
            name += "-" + str(self.num)
        with file_logger.FileLogger(self.config.path, name, self.config.quiet) as logger:
            print("{} started".format(name))
            logger.print("{} started".format(name))
            try:
                return self.run(*args, **kwargs)
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
