import abc
import itertools
import traceback

import numpy as np
from open_spiel.python.algorithms.alpha_zero_mpg.utils import Buffer
from open_spiel.python.utils import spawn
import open_spiel.python.algorithms.mcts as mcts

from open_spiel.python.algorithms.alpha_zero_mpg.mcts import evaluator as mcts_evaluator, guide as mcts_guide
from open_spiel.python.algorithms.alpha_zero_mpg import resource,utils,bots as bots_lib


class Evaluator(utils.Watched):
    def __init__(self, config, num=None, name=None):
        super().__init__(config, num, name)
        pass


class MultiProcEvaluator(Evaluator):
    def __init__(self, config, num=None,opponent="mcts", name=None):
        super().__init__(config, num, name)
        self.opponent=opponent
        pass

    def run(self, logger, queue, game):
        config = self.config
        results = Buffer(self.config.evaluation_window)
        logger.print("Initializing model")
        model = resource.SavedModelBundle(logger, config, game)
        logger.print("Initializing bots")
        guide=mcts_guide.LinearGuide(t=0.3,payoffs_prior=mcts_guide.PayoffsSoftmaxPrior())
        az_evaluator = mcts_evaluator.GuidedAlphaZeroEvaluator(game, model,guide)
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
            is_updated=model.update(path, az_evaluator)
            if is_updated:
                logger.print("Updated model, new hash is {}.".format(model.hash()))
            # Alternate between playing as player Max and player Min.
            az_player = game_num % 2
            # Each difficulty is repeated twice, once per player. Hence, the Euclidean division by 2.
            difficulty = (game_num // 2) % config.eval_levels
            max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))

            az_bot=bots_lib.init_bot(config=config, game=game, evaluator_=az_evaluator, evaluation=True,bot_type="alpha-zero")
            if self.opponent=="mcts":
                opponent_bot=bots_lib.init_bot(config=config, game=game, evaluator_=random_evaluator, evaluation=True,
                                    uct_c=config.uct_c,max_simulations=max_simulations,solve=True,
                                    verbose=False,dont_return_chance_node=True, bot_type="mcts")
            else:
                opponent_bot=bots_lib.init_bot(config=config, game=game, evaluator_=az_evaluator, evaluation=True,bot_type=self.opponent,
                                               player_id=1-az_player)

            bots = [
                az_bot,
                opponent_bot
            ]
            if az_player == 1:
                bots = list(reversed(bots))

            trajectory = utils.play_game(logger, game_num, game, bots, temperature=1,
                                         temperature_drop=0, fix_environment=config.fix_environment)
            results.append(trajectory.returns[az_player])
            queue.put((difficulty, trajectory.returns[az_player]))

            logger.print("AZ: {}, {}: {}, AZ avg/{}: {:.3f}".format(
                trajectory.returns[az_player],
                self.opponent.upper(),
                trajectory.returns[1 - az_player],
                len(results), np.mean(results.data)))

    def __call__(self, logger, queue, game):
        return self.run(logger, queue, game)


class NetworkEvaluator(Evaluator):
    pass
