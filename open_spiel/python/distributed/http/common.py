import argparse
import os
import socket
import sys
from enum import Enum

import fastapi
import pyspiel
import yaml
from open_spiel.python.algorithms.alpha_zero_mpg.utils import nested_dict_to_namespace
import open_spiel.python.algorithms.alpha_zero_mpg.utils as mpg_utils


def make_discovery_directory(config):
    discovery_dir = os.path.join(config.path, "services")
    os.makedirs(discovery_dir, exist_ok=True)
    for service in ["actor", "learner", "evaluator"]:
        service_dir = os.path.join(discovery_dir, service)
        os.makedirs(service_dir, exist_ok=True)
    return discovery_dir


class AlphaZeroService(fastapi.FastAPI):
    def __init__(self, service_type: str, path=None):
        super().__init__()
        if service_type not in ["actor", "learner", "evaluator"]:
            raise ValueError(f"Invalid service type `{service_type}`")
        self.config = None
        argv = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="config.yaml")
        args, _ = parser.parse_known_args(argv)
        if hasattr(args, "config") and os.path.exists(args.config):
            with open(args.config, "r") as f:
                self.config = yaml.safe_load(f)
        elif os.path.exists("config.yaml"):
            with open("config.yaml", "r") as f:
                self.config = yaml.safe_load(f)
        elif path is not None:
            with open(path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("No config found")

        self.config = nested_dict_to_namespace(self.config)
        config = self.config
        mpg_utils.compatibility_mode(self.config)
        complete_game = mpg_utils.game_complete_name(config.game)
        game = pyspiel.load_game(*complete_game)
        if game.observation_tensor_shape_specs() == pyspiel.TensorShapeSpecs.VECTOR:
            shape = game.observation_tensor_shape()
        else:
            shape = game.observation_tensor_shapes_list()
        config.observation_shape = shape
        config.output_size = game.num_distinct_actions()
        self.game = game
        make_discovery_directory(config)
        self.working_directory = os.path.join(self.services_path, service_type,socket.gethostname())
        os.makedirs(self.working_directory, exist_ok=True)
        config.working_directory = self.working_directory

    @property
    def services_path(self):
        return os.path.join(self.config.path, "services")


def format_date(date):
    return date.strftime("%Y-%m-%d %H:%M:%S")
