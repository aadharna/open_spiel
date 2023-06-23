import sys
from typing import Union

import fastapi
import open_spiel.python.algorithms.alpha_zero_mpg.services.actor as actor
import open_spiel.python.algorithms.alpha_zero_mpg.dto as mpg_dto
import yaml
import argparse
import os
from .. import common


class EvaluatorApp(common.AlphaZeroService):
    pass


app = EvaluatorApp()


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


@app.get("/discovery")
def discovery():
    return NotImplementedError()