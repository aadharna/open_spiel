from typing import TypedDict

import numpy as np
import open_spiel.python.algorithms.mcts as mcts
import open_spiel.python.bots.uniform_random as uniform_random
import pyspiel

REGISTERED_BOTS = {}


def register_bot(name):
    def _register(func):
        global REGISTERED_BOTS
        if name in REGISTERED_BOTS:
            raise ValueError("Bot with name {} already registered".format(name))

        REGISTERED_BOTS[name] = func
        return func

    return _register


@register_bot("greedy")
def init_greedy_bot(config, game, evaluator_, evaluation, player_id, seed,**kwargs):
    """Initializes a bot."""
    return pyspiel.mpg.GreedyBot()


@register_bot("eps-greedy")
def init_eps_greedy_bot(config, game, evaluator_, evaluation, player_id, seed,*,epsilon=0.1,**kwargs):
    """Initializes a bot."""
    return pyspiel.mpg.EpsilonGreedyBot


def init_mcts_bot_general(cls,config, game, evaluator_, evaluation,uct_c=None,max_simulations=None,solve=False,**kwargs):
    noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
    if uct_c is None:
        uct_c = config.uct_c
    if max_simulations is None:
        max_simulations = config.max_simulations
    return cls(
        game,
        uct_c,
        max_simulations,
        evaluator_,
        solve=solve,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True)

@register_bot("py-mcts")
def init_mcts_bot(config, game, evaluator_, evaluation,uct_c=None,max_simulations=None,solve=False,**kwargs):
    """Initializes a bot."""
    return init_mcts_bot_general(mcts.MCTSBot,config, game, evaluator_, evaluation,
                                 uct_c=uct_c, max_simulations=max_simulations, solve=solve,**kwargs)


@register_bot("mcts")
def init_mcts_bot_default(config, game, evaluator_, evaluation,uct_c=None,max_simulations=None,solve=False,**kwargs):
    return init_mcts_bot(config, game, evaluator_, evaluation,uct_c=uct_c, max_simulations=max_simulations, solve=solve,**kwargs)


@register_bot("cpp-mcts")
def init_mcts_cpp_bot(config, game, evaluator_, evaluation,uct_c=None,max_simulations=None,solve=False,**kwargs):
    """Initializes a bot."""
    return init_mcts_bot_general(pyspiel.MCTSBot, config=config, game=game, evaluator_=evaluator_, evaluation=evaluation,
                                 uct_c=uct_c, max_simulations=max_simulations, solve=solve,**kwargs)


@register_bot("random")
def init_random_bot(player_id, seed=None,**kwargs):
    """Initializes a bot."""
    return uniform_random.UniformRandomBot(player_id, np.random.RandomState(seed))


@register_bot("alpha-zero")
def init_alpha_zero_bot(config, game, evaluator_, evaluation,**kwargs):
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


def init_bot(config, game, evaluator_, evaluation, bot_type, player_id=None, seed=None, **kwargs):
    if seed is None:
        seed = np.random.randint(2 ** 32 - 1)
    rng = np.random.RandomState(seed)

    if bot_type not in REGISTERED_BOTS:
        raise ValueError("Unknown bot type: {}".format(bot_type))
    return REGISTERED_BOTS[bot_type](config=config, game=game, evaluator_=evaluator_, evaluation=evaluation,
                                     player_id=player_id, seed=seed, **kwargs)

