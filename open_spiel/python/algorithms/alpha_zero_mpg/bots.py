from typing import TypedDict

import numpy as np
import open_spiel.python.algorithms.mcts as mcts
import open_spiel.python.bots.uniform_random as uniform_random
import pyspiel


def register_bot_class(name):
    def _register(cls):
        if name in REGISTERED_BOTS:
            raise ValueError("Bot with name {} already registered".format(name))
        REGISTERED_BOTS[name] = cls
        return cls

    return _register


def register_bot(name):
    def _register(func):
        if name in REGISTERED_BOTS:
            raise ValueError("Bot with name {} already registered".format(name))

        REGISTERED_BOTS[name] = func
        return func

    return _register


@register_bot("greedy")
class GreedyBot(pyspiel.Bot):
    def __init__(self, player_id, rng):
        """Initializes a uniform-random bot.

        Args:
          player_id: The integer id of the player for this bot, e.g. `0` if acting
            as the first player.
          rng: A random number generator supporting a `choice` method, e.g.
            `np.random`
        """
        pyspiel.Bot.__init__(self)
        self._player_id = player_id
        self._rng = rng

    def restart_at(self, state):
        pass

    def player_id(self):
        return self._player_id

    def provides_policy(self):
        return True

    def step_with_policy(self, state):
        """Returns the greedy policy and selected action in the given state.

        Args:
          state: The current state of the game.

        Returns:
          A `(policy, action)` pair, where policy is a `list` of
          `(action, probability)` pairs for each legal action, with
          `probability = 1/num_actions`
          The `action` is selected uniformly at random from the legal actions,
          or `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        legal_actions = state.legal_actions(self._player_id)
        if not legal_actions:
            return [], pyspiel.INVALID_ACTION
        p = 1 / len(legal_actions)
        policy = [(action, p) for action in legal_actions]
        action = self._rng.choice(legal_actions)
        return policy, action

    def step(self, state):
        return self.step_with_policy(state)[1]


REGISTERED_BOTS = {}


@register_bot("mcts")
def init_mcts_bot(config, game, evaluator_, evaluation, player_id=None, seed=None):
    """Initializes a bot."""
    noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
    return mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        evaluator_,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True)


@register_bot("random")
def init_random_bot(config, game, evaluator_, evaluation, player_id=None, seed=None):
    """Initializes a bot."""
    return uniform_random.UniformRandomBot(player_id, np.random.RandomState(seed))


def init_bot(config, game, evaluator_, evaluation, bot_type=None, player_id=None, seed=None, **kwargs):
    if seed is None:
        seed = np.random.randint(2 ** 32 - 1)
    rng = np.random.RandomState(seed)

    if bot_type is None:
        bot_type = config.bot_type
    if bot_type not in REGISTERED_BOTS:
        raise ValueError("Unknown bot type: {}".format(bot_type))
    return REGISTERED_BOTS[bot_type](config, game, evaluator_, evaluation, player_id, seed, **kwargs)
