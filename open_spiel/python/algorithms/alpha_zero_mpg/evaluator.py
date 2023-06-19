import abc
import itertools
import traceback

import numpy as np
from open_spiel.python.algorithms.alpha_zero.alpha_zero_v2 import Buffer
from open_spiel.python.utils import spawn
import open_spiel.python.algorithms.mcts as mcts

from .mcts import evaluator as evaluator_lib
from . import resource,utils,bots as bots_lib


class Evaluator(utils.Watched):
    def __init__(self, config, num=None, name=None):
        super().__init__(config, num, name)
        pass


class MultiProcEvaluator(Evaluator):
    def __init__(self, config, num=None, name=None):
        super().__init__(config, num, name)
        pass

    def run(self, logger, queue, game):
        config = self.config
        results = Buffer(self.config.evaluation_window)
        logger.print("Initializing model")
        model = resource.SavedModelBundle(logger, config, game)
        logger.print("Initializing bots")
        az_evaluator = evaluator_lib.MPGAlphaZeroEvaluator(game, model)
        random_evaluator = mcts.RandomRolloutEvaluator()

        for game_num in itertools.count():
            path = None
            while True:
                try:
                    path = queue.get_nowait()
                except spawn.Empty:
                    break
            if path == "":
                return
            model.update(path, az_evaluator)
            # Alternate between playing as player Max and player Min.
            az_player = game_num % 2
            # Each difficulty is repeated twice, once per player. Hence, the Euclidean division by 2.
            difficulty = (game_num // 2) % config.eval_levels
            max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
            bots = [
                bots_lib.init_bot(config=config, game=game, evaluator_=az_evaluator, evaluation=True,bot_type="alpha-zero"),
                mcts.MCTSBot(
                    game,
                    config.uct_c,
                    max_simulations,
                    random_evaluator,
                    solve=True,
                    verbose=False,
                    dont_return_chance_node=True)
            ]
            if az_player == 1:
                bots = list(reversed(bots))

            trajectory = utils.play_game(logger, game_num, game, bots, temperature=1,
                                         temperature_drop=0, fix_environment=config.fix_environment)
            results.append(trajectory.returns[az_player])
            queue.put((difficulty, trajectory.returns[az_player]))

            logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
                trajectory.returns[az_player],
                trajectory.returns[1 - az_player],
                len(results), np.mean(results.data)))

    def __call__(self, logger, queue, game):
        return self.run(logger, queue, game)


class NetworkEvaluator(Evaluator):
    pass
