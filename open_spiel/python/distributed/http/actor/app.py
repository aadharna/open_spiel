import sys
from typing import Union

import fastapi
import open_spiel.python.algorithms.alpha_zero_mpg.services.actor as actor
import open_spiel.python.algorithms.alpha_zero_mpg.dto as mpg_dto
import yaml
import argparse
import os


class ActorApp(fastapi.FastAPI):
    def __init__(self):
        super().__init__()
        self.config = None
        argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="config.yaml")
        args, _ = parser.parse_known_args(argv)
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                self.config = yaml.safe_load(f)


app = ActorApp()


@app.get("/config")
def get_config():
    return app.config

@app.post("/start")
def start(config,name: Union[str,None]=None, num: Union[int,None]=None):
    try:
        app.actor = actor.MultiProcActor(config, num=num, name=name)
    except:
        