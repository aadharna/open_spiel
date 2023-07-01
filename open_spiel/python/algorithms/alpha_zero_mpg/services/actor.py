import abc
import itertools
import traceback
import datetime
from typing import TypedDict, Union, List

import numpy as np

from ..multiprocess import queue as queue_lib

from open_spiel.python.algorithms.alpha_zero_mpg import utils
from open_spiel.python.utils import spawn

from open_spiel.python.algorithms.alpha_zero_mpg.mcts import evaluator as mcts_evaluator
from open_spiel.python.algorithms.alpha_zero_mpg.mcts import guide as mcts_guide
from open_spiel.python.algorithms.alpha_zero_mpg import resource, bots as bots_lib


class SelfPlayingGameStats(TypedDict):
    winner: Union[str, None]
    num_steps: int
    graph_size: int
    edges_count: int



class ActorStats(TypedDict):
    counter: Union[int,None]
    start_time: datetime.datetime
    end_time: datetime.datetime
    num_games: int
    games: List[SelfPlayingGameStats]
    pass


class Actor(utils.Watched):

    def __init__(self, config, num=None, name=None, **kwargs):
        super().__init__(config, num, name, **kwargs)

        pass


class MultiProcActor(Actor):
    def __init__(self, config, num=None, name=None,seed=None, **kwargs):
        super().__init__(config, num, name, **kwargs)
        # Time resolution in seconds for updating the actor's state.
        self.stats_frequency = config.services.actors.stats_frequency or config.stats_frequency or 60
        self._stats = None
        self.rng=np.random.RandomState(seed)
        pass

    @property
    def stats(self) -> Union[None, ActorStats]:
        return self._stats

    def _reset_stats(self):
        self._stats = ActorStats(start_time=datetime.datetime.now(), end_time=datetime.datetime.now(), num_games=0,
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
        logger.print("Initializing model")
        model = resource.SavedModelBundle(logger, config, game)
        logger.print("Initializing bots")
        guide = mcts_guide.LinearGuide(t=0.3, payoffs_prior=mcts_guide.PayoffsSoftmaxPrior())
        #az_evaluator = mcts_evaluator.GuidedAlphaZeroEvaluator(game, model, guide)
        az_evaluator = mcts_evaluator.AlphaZeroEvaluator(game, model)
        bots = [
            bots_lib.init_bot(config=config.mcts, game=game, evaluator_=az_evaluator, evaluation=False,
                              bot_type="alpha-zero"),
            bots_lib.init_bot(config=config.mcts, game=game, evaluator_=az_evaluator, evaluation=False,
                              bot_type="alpha-zero")
        ]
        payoff_offset=utils.PayoffOffset.from_registry(name=config.replay_buffer.payoff_offset.distribution,params=config.replay_buffer.payoff_offset.params)

        self._reset_stats()
        for game_num in itertools.count():
            if self._stats["counter"] is None:
                self._stats["counter"] = game_num

            self._stats["end_time"] = datetime.datetime.now()
            if (self._stats["end_time"] - self._stats["start_time"]).seconds > self.stats_frequency:
                queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_ANALYSIS, self.stats))
                self._reset_stats()

            path = None
            while True:
                try:
                    message_type, message = queue.get_nowait()
                    if message_type == queue_lib.MessageTypes.QUEUE_MESSAGE:
                        path=message
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
            is_updated = model.update(path, az_evaluator)
            if is_updated:
                logger.print("Updated model, new hash is {}.".format(model.hash()))

            message_data = utils.play_game(logger, game_num, game, bots, config.temperature,
                                           config.temperature_drop, fix_environment=config.fix_environment,payoff_offset=payoff_offset)
            self._stats["num_games"] += 1
            game_stats: SelfPlayingGameStats = {
                "winner": utils.get_winner_name(message_data.returns[0]),
                "num_steps": len(message_data.states),
                "graph_size": message_data.graph_size,
                "edges_count": message_data.edges_count
                                     }
            self._stats["games"].append(game_stats)
            queue.put(queue_lib.QueueMessage(queue_lib.MessageTypes.QUEUE_MESSAGE, message_data))
        # Close the queue

    def __call__(self, logger, queue, game):
        return self.run(logger, queue, game)
