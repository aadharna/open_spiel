import abc
import datetime
import itertools
import traceback
from typing import TypedDict, Union, List
from ..multiprocess import queue as queue_lib
import numpy as np
from open_spiel.python.algorithms.alpha_zero_mpg.utils import Buffer
from open_spiel.python.utils import spawn
import open_spiel.python.algorithms.mcts as mcts

from open_spiel.python.algorithms.alpha_zero_mpg.mcts import evaluator as mcts_evaluator, guide as mcts_guide
from open_spiel.python.algorithms.alpha_zero_mpg import resource, utils, bots as bots_lib


class EvaluationGameStats(TypedDict):
    winner: Union[int, None]
    num_steps: int
    start_time: datetime.datetime
    end_time: datetime.datetime
    player_max: str
    player_min: str
    # The relative difficulty of the games. This is ratio between the maximum simulation of the opponent with respect to alpha zerp.
    # None if the opponent is not mcts.
    relative_difficulty: Union[float, None]


class EvaluationStats(TypedDict):
    counter: Union[int, None]
    start_time: datetime.datetime
    end_time: datetime.datetime
    num_games: int
    games: List[EvaluationGameStats]
    pass


class Evaluator(utils.Watched):
    def __init__(self, config, num=None, name=None, **kwargs):
        super().__init__(config, num, name, **kwargs)
        pass


class MultiProcEvaluator(Evaluator):
    def __init__(self, config, num=None, opponent="mcts", name=None, **kwargs):
        super().__init__(config, num, name, **kwargs)
        self.opponent = opponent
        self.stats_frequency = config.services.evaluators.stats_frequency or config.stats_frequency or 60
        self._stats = None

        pass

    @property
    def stats(self) -> Union[None, EvaluationStats]:
        return self._stats

    def _reset_stats(self):
        self._stats = EvaluationStats(start_time=datetime.datetime.now(), end_time=datetime.datetime.now(), num_games=0,
                                      games=[], counter=None)

    def _sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def run(self, logger, queue, game):
        config = self.config
        results = Buffer(self.config.evaluation_window)
        logger.print("Initializing model")
        model = resource.SavedModelBundle(logger, config, game)
        logger.print("Initializing bots")
        guide = mcts_guide.LinearGuide(t=0.3, payoffs_prior=mcts_guide.PayoffsSoftmaxPrior())
        az_evaluator = mcts_evaluator.GuidedAlphaZeroEvaluator(game, model, guide)
        random_evaluator = mcts.RandomRolloutEvaluator()
        self._reset_stats()
        for game_num in itertools.count():
            if self._stats["counter"] is None:
                self._stats["counter"] = game_num

            if (self._stats["end_time"] - self._stats["start_time"]).seconds > self.stats_frequency:
                queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_ANALYSIS, self.stats))
                self._reset_stats()

            path = None
            while True:
                try:
                    message_type, message = queue.get_nowait()
                    if message_type == queue_lib.MessageTypes.QUEUE_MESSAGE:
                        path= message
                    elif message_type == queue_lib.MessageTypes.QUEUE_HEARTBEAT:
                        queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_HEARTBEAT, None))
                    elif message_type == queue_lib.MessageTypes.QUEUE_ANALYSIS:
                        queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_ANALYSIS, self.stats))
                        pass
                    elif message_type == queue_lib.MessageTypes.QUEUE_CLOSE:
                        close_message = queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_CLOSE, None)
                        queue.put(close_message)
                        return
                except spawn.Empty:
                    break
            if path == "":
                return
            is_updated = model.update(path, az_evaluator)
            if is_updated:
                logger.print("Updated model, new hash is {}.".format(model.hash()))
            # Alternate between playing as player Max and player Min.
            az_player = game_num % 2
            # Each difficulty is repeated twice, once per player. Hence, the Euclidean division by 2.
            difficulty = (game_num // 2) % config.services.evaluators.evaluation_levels
            max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))

            az_bot = bots_lib.init_bot(config=config.mcts, game=game, evaluator_=az_evaluator, evaluation=True,
                                       bot_type="alpha-zero")

            game_stats = {}
            bot_names = ["alpha-zero", self.opponent]
            if self.opponent == "mcts":
                game_stats["relative_difficulty"] = max_simulations / config.max_simulations
                opponent_bot = bots_lib.init_bot(config=config.mcts, game=game, evaluator_=random_evaluator,
                                                 evaluation=True,
                                                 uct_c=config.uct_c, max_simulations=max_simulations, solve=True,
                                                 verbose=False, dont_return_chance_node=True, bot_type="mcts")
            else:
                game_stats["relative_difficulty"] = None
                opponent_bot = bots_lib.init_bot(config=config.mcts, game=game, evaluator_=az_evaluator,
                                                 evaluation=True, bot_type=self.opponent,
                                                 player_id=1 - az_player)

            bots = [
                az_bot,
                opponent_bot
            ]
            if az_player == 1:
                bots = list(reversed(bots))
                bot_names = list(reversed(bot_names))

            game_stats["player_max"] = bot_names[0]
            game_stats["player_min"] = bot_names[1]

            start_time = datetime.datetime.now()
            trajectory = utils.play_game(logger, game_num, game, bots, temperature=1,
                                         temperature_drop=0, fix_environment=config.fix_environment)
            end_time = datetime.datetime.now()
            # Set game time stats
            game_stats["start_time"] = start_time
            game_stats["end_time"] = end_time

            # Set game results
            results.append(trajectory.returns[az_player])

            # Send results back to the orchestrator
            queue.put((difficulty, trajectory.returns[az_player]))

            # Update stats
            game_stats["winner"] = self._sign(trajectory.returns[0])
            game_stats["num_steps"] = len(trajectory.states)
            self._stats["games"].append(EvaluationGameStats(**game_stats))
            self._stats["num_games"] += 1

            # Log results
            logger.print("AZ: {}, {}: {}, AZ avg/{}: {:.3f}".format(
                trajectory.returns[az_player],
                self.opponent.upper(),
                trajectory.returns[1 - az_player],
                len(results), np.mean(results.data)))

    def __call__(self, logger, queue, game):
        return self.run(logger, queue, game)


class NetworkEvaluator(Evaluator):
    pass
