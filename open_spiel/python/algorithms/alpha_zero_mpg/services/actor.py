import abc
import itertools
import traceback

from open_spiel.python.algorithms.alpha_zero_mpg import utils
from open_spiel.python.utils import spawn

from open_spiel.python.algorithms.alpha_zero_mpg.mcts import evaluator as mcts_evaluator
from open_spiel.python.algorithms.alpha_zero_mpg.mcts import guide as mcts_guide
from open_spiel.python.algorithms.alpha_zero_mpg import resource,bots as bots_lib


class Actor(utils.Watched):
    def __init__(self, config, num=None, name=None):
        super().__init__(config, num, name)

        pass


class MultiProcActor(Actor):
    def __init__(self, config, num=None, name=None):
        super().__init__(config, num, name)

        pass

    def run(self, logger, queue, game):
        config = self.config
        logger.print("Initializing model")
        model = resource.SavedModelBundle(logger, config, game)
        logger.print("Initializing bots")
        guide=mcts_guide.LinearGuide(t=0.3,payoffs_prior=mcts_guide.PayoffsSoftmaxPrior())
        az_evaluator = mcts_evaluator.GuidedAlphaZeroEvaluator(game, model,guide)
        bots = [
            bots_lib.init_bot(config=config, game=game, evaluator_=az_evaluator, evaluation=False,
                              bot_type="alpha-zero"),
            bots_lib.init_bot(config=config, game=game, evaluator_=az_evaluator, evaluation=False,
                              bot_type="alpha-zero")
        ]
        for game_num in itertools.count():
            path = None
            while True:
                try:
                    path = queue.get_nowait()
                except spawn.Empty:
                    break
            if path == "":
                return
            is_updated= model.update(path, az_evaluator)
            if is_updated:
                logger.print("Updated model, new hash is {}.".format(model.hash()))

            queue.put(utils.play_game(logger, game_num, game, bots, config.temperature,
                                      config.temperature_drop, fix_environment=config.fix_environment))
    def __call__(self, logger, queue, game):
        return self.run(logger, queue, game)

class NetworkActor(Actor):
    pass
