import abc
import itertools
import traceback

from open_spiel.open_spiel.python.algorithms.alpha_zero_mpg import utils
from open_spiel.open_spiel.python.algorithms.alpha_zero_mpg.alpha_zero import _init_bot
from open_spiel.open_spiel.python.algorithms.alpha_zero_mpg.utils import file_logger
from open_spiel.open_spiel.python.algorithms.alpha_zero_mpg.alpha_zero import _init_model_from_config, _play_game
from open_spiel.python.utils import spawn

from mcts import evaluator as evaluator_lib
from . import resource


class Actor(utils.Watched):
    def __init__(self, config, num=None, name=None):
        super().__init__(config, num, name)

        pass

    def update_model(self):
        pass

    def run(self, logger, queue, game):
        config = self.config
        logger.print("Initializing model")
        model = resource.SavedModelBundle(logger, config, game)
        logger.print("Initializing bots")
        az_evaluator = evaluator_lib.MPGAlphaZeroEvaluator(game, model)
        bots = [
            _init_bot(config, game, az_evaluator, False),
            _init_bot(config, game, az_evaluator, False),
        ]
        for game_num in itertools.count():
            path=None
            while True:
                try:
                    path=queue.get_nowait()
                except spawn.Empty:
                    break
            if path=="":
                return
            model.update(path, az_evaluator)
            queue.put(_play_game(logger, game_num, game, bots, config.temperature,
                                 config.temperature_drop, fix_environment=config.fix_environment))
