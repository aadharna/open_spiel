import argparse
import datetime
import json
import multiprocessing
import threading
import os
import sys
import tempfile
from typing import Union, Tuple, Iterable

from . import utils

import tensorflow as tf

import pyspiel
from open_spiel.python.utils import spawn

from open_spiel.python.algorithms.alpha_zero_mpg import dto
from open_spiel.python.algorithms.alpha_zero_mpg.multiprocess import orchestrator

JOIN_WAIT_DELAY = 0.001

from open_spiel.python.algorithms.alpha_zero_mpg.utils import Config
from open_spiel.python.algorithms.alpha_zero_mpg.services import learner, evaluator, actor
import reverb


class LocalActorFactory:
    def __init__(self, specs):
        self.specs = specs
        pass

    def __call__(self, config, game, queue, num):
        act = actor.MultiProcActor(config, num, name="actor")
        return act.start(queue=queue, game=game)


class LocalEvaluatorFactory:
    def __init__(self, specs):
        self.specs = specs
        pass

    def __call__(self, config, game, queue, num, opponent="mcts"):
        eval = evaluator.MultiProcEvaluator(config, num, name="evaluator", opponent=opponent)
        return eval.start(queue=queue, game=game)


def create_local_actor(*, game, config, queue, num):
    """Create a local actor process."""
    act = actor.MultiProcActor(config, num, name="actor")
    return act.start(queue=queue, game=game)



EXIT_FUNCTIONS = []


def register_exit_function(fn, *args, **kwargs):
    EXIT_FUNCTIONS.append((fn, args, kwargs))


def exit_functions():
    for fn, args, kwargs in EXIT_FUNCTIONS:
        fn(*args, **kwargs)


def alpha_zero(config: Config):
    """Start all the worker processes for a full alphazero setup."""
    complete_game = utils.game_complete_name(config.game)
    game = pyspiel.load_game(*complete_game)
    if game.observation_tensor_shape_specs() == pyspiel.TensorShapeSpecs.VECTOR:
        shape = game.observation_tensor_shape()
    else:
        shape = game.observation_tensor_shapes_list()
    config.observation_shape = shape
    config.output_size = game.num_distinct_actions()

    print("Starting game", complete_game)
    if game.num_players() != 2:
        sys.exit("AlphaZero can on²ly handle 2-player games.")
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
        raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must have sequential turns.")
    if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("Game must be deterministic.")

    if config.game.fix_environment:
        print("Fixing environment:")
        print(game.new_initial_state())

    path = config.path
    if not path:
        path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), game.get_type().short_name))
        config.path = path

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        sys.exit("{} isn't a directory".format(path))
    print("Writing logs and checkpoints to:", path)
    print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                      config.nn_depth))

    with open(os.path.join(config.path, "config.json"), "w") as fp:
        config_json = utils.recursive_namespace_todict(config)
        fp.write(json.dumps(config_json, indent=2, sort_keys=True) + "\n")

    actor_factory = LocalActorFactory(config)
    evaluator_factory = LocalEvaluatorFactory(config)
    if isinstance(config.services.actors.instances, int):
        actors = [spawn.Process(actor_factory, kwargs={"game": game, "config": config,
                                                       "num": i})
                  for i in range(config.services.actors.instances)
                  ]
    else:
        raise ValueError("As of now, only int is supported for actors")

    if isinstance(config.services.evaluators.instances, int):
        evaluators = [spawn.Process(evaluator_factory, kwargs={"game": game, "config": config,
                                                               "num": i, "opponent": "random"})
                      for i in range(config.services.evaluators.instances)]
    else:
        raise ValueError("As of now, only int is supported for evaluators")

    try:
        model_broadcaster = dto.ProcessesBroadcaster(actors + evaluators)
        if config.replay_buffer.implementation.type == "grpc":
            table = reverb.Table(
                name=config.replay_buffer.implementation.table,
                sampler=utils.expand_arguments(utils.get_reverb_selector(config.replay_buffer.implementation.sampler.name)),
                remover=utils.expand_arguments(utils.get_reverb_selector(config.replay_buffer.implementation.remover.name)),
                max_size=config.replay_buffer.buffer_size,
                rate_limiter=reverb.rate_limiters.MinSize(config.replay_buffer.implementation.min_size),
                signature=utils.get_reverb_signature()
            )
            reverb_server = reverb.Server(tables=[table], port=config.replay_buffer.implementation.port)
            replay_buffer = dto.GrpcReplayBuffer(config=config.replay_buffer.implementation,batch_size=config.training.batch_size,
                                                 padding=config.training.padding)
            actors_orchestrator = orchestrator.ActorsGrpcOrchestrator(actors, config=config,
                                                                      max_game_length=game.max_game_length())
            # evaluator_orchestrator = orchestrator.EvaluatorOrchestrator(evaluators)
            register_exit_function(reverb_server.stop)
            register_exit_function(actors_orchestrator.stop)
            register_exit_function(model_broadcaster.request_exit)


        elif config.reverb_buffer.implementation.type == "queues":
            replay_buffer = dto.QueuesReplayBuffer(config.replay_buffer_size, config.replay_buffer_reuse,
                                                   actors=actors, max_game_length=game.max_game_length())
        else:
            raise ValueError(f"Unknown replay buffer implementation type {config.reverb_buffer.implementation.type}")
        adapter = learner.Learner(config=config, replay_buffer=replay_buffer, model_broadcaster=model_broadcaster)
        adapter.start(game)
    except (KeyboardInterrupt, EOFError):
        print("Caught a KeyboardInterrupt, stopping early.")
    finally:
        print("Stopping actors and evaluators")
        exit_functions()

        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        # for proc in actors:
        #    while proc.exitcode is None:
        #        while not proc.queue.empty():
        #            proc.queue.get_nowait()
        #        proc.join(JOIN_WAIT_DELAY)
        # for proc in evaluators:
        #    proc.join()
