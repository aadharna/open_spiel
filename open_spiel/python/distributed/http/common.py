import argparse
import os
import sys

import fastapi
import yaml


class AlphaZeroService(fastapi.FastAPI):
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

