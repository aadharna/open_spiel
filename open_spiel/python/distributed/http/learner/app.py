import sys
from typing import Union

import fastapi
import open_spiel.python.algorithms.alpha_zero_mpg.services.actor as actor
import open_spiel.python.algorithms.alpha_zero_mpg.dto as mpg_dto
import open_spiel.python.algorithms.alpha_zero_mpg.utils as utils
import reverb
import tensorflow as tf

from .. import common
from open_spiel.python.algorithms.alpha_zero_mpg.services import learner
import pyspiel


class LearnerApp(common.AlphaZeroService):

    def __init__(self):
        super().__init__()
        if self.config.replay_buffer.implementation.name != "reverb":
            raise ValueError("Only reverb replay buffer is supported on server side")
        table = reverb.Table(
            name=self.config.replay_buffer.implementation.table,
            sampler=utils.expand_arguments(utils.get_reverb_selector(self.config.replay_buffer.implementation.sampler.name)),
            remover=utils.expand_arguments(utils.get_reverb_selector(self.config.replay_buffer.implementation.remover.name)),
            max_size=self.config.replay_buffer.buffer_size,
            rate_limiter=reverb.rate_limiters.MinSize(self.config.replay_buffer.implementation.min_size),
            signature=utils.get_reverb_signature()
        )
        self.reverb_server = reverb.Server(tables=[table], port=self.config.replay_buffer.implementation.port)
        self.replay_buffer = mpg_dto.GrpcReplayBuffer(address=self.config.replay_buffer.implementation.address,
                                             port=self.config.replay_buffer.implementation.port,
                                             table=self.config.replay_buffer.implementation.table)

        self.game= pyspiel.load_game(self.config.game.name)
        self.learner = learner.Learner(config=self.config, replay_buffer=self.replay_buffer, model_broadcaster=model_broadcaster)
        self.learner.start(self.game)
    pass


app = LearnerApp()


@app.get("/config")
def get_config():
    return app.config

@app.post("/start")
def start(config,name: Union[str,None]=None, num: Union[int,None]=None):
    app.actor = actor.MultiProcActor(config, num=num, name=name)

@app.get("/stop")
def stop():
    raise NotImplementedError()

@app.get("/health")
def health():
    return True

@app.get("/stats")
def stats():
    return NotImplementedError()


@app.get("/update")
def update():
    return NotImplementedError()
