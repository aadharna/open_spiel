import argparse
import datetime
import json
import multiprocessing
import threading
import os
import sys
import tempfile
from typing import Union, Tuple,Iterable

import tensorflow as tf

import pyspiel
from open_spiel.python.utils import spawn

from open_spiel.python.algorithms.alpha_zero_mpg import dto

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


def recursive_namespace_todict(ns):
    """Converts a namespace object to a dictionary, recursively."""
    d = {}
    if isinstance(ns, argparse.Namespace):
        ns=vars(ns)

    if isinstance(ns, dict):
        return {k: recursive_namespace_todict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [recursive_namespace_todict(v) for v in ns]
    else:
        return ns


def reduce_lists(ns):
    result={}
    if isinstance(ns,dict):
        for k,v in ns.items():
            r=reduce_lists(v)
            result[k]=r
        return result
    if isinstance(ns, list):
        mapper=map(str,ns)
        return " ".join(mapper)
    return ns


def game_complete_name(game_config) -> Union[Tuple[str], Tuple[str, str]]:
    game_dict = reduce_lists(recursive_namespace_todict(game_config))
    if "generator" in game_dict:
        game_dict["generator_params"]=game_dict["generator"]["params"]
        game_dict["generator"]=game_config.generator.name
    game_params=[]
    game_name=game_dict["name"]
    game_dict.pop("name")
    game_dict.pop("fix_environment")
    if len(game_dict) == 0:
        return (game_name,)
    else:
        return (game_name, game_dict)

def get_selector(name):
    if name == "random" or name == "uniform":
        return reverb.selectors.Uniform
    elif name == "fifo":
        return reverb.selectors.Fifo
    elif name == "priority":
        return reverb.selectors.Prioritized
    elif name == "lifo":
        return reverb.selectors.Lifo
    else:
        raise ValueError("Unknown selector: {}".format(name))


def expand_arguments(fn,*args,**kwargs):
    if len(args) == 1:
        args=args[0]
    if isinstance(args,argparse.Namespace) or isinstance(args,dict):
        args=vars(args)
        return fn(**args, **kwargs)
    elif isinstance(args,Iterable):
        return fn(*args, **kwargs)
    else:
        return  fn(args, **kwargs)

def alpha_zero(config: Config):
    """Start all the worker processes for a full alphazero setup."""
    complete_game = game_complete_name(config.game)
    game = pyspiel.load_game(*complete_game)
    if game.observation_tensor_shape_specs() == pyspiel.TensorShapeSpecs.VECTOR:
        shape = game.observation_tensor_shape()
    else:
        shape = game.observation_tensor_shapes_list()
    config.observation_shape = shape
    config.output_size = game.num_distinct_actions()

    print("Starting game", complete_game)
    if game.num_players() != 2:
        sys.exit("AlphaZero can only handle 2-player games.")
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
        config_json=recursive_namespace_todict(config)
        fp.write(json.dumps(config_json, indent=2, sort_keys=True) + "\n")

    actor_factory = LocalActorFactory(config)
    evaluator_factory = LocalEvaluatorFactory(config)
    if isinstance(config.services.actors, int):
        actors = [spawn.Process(actor_factory, kwargs={"game": game, "config": config,
                                                       "num": i})
                  for i in range(config.services.actors)
        ]
    else:
        raise ValueError("As of now, only int is supported for actors")

    if isinstance(config.services.evaluators, int):
        evaluators = [spawn.Process(evaluator_factory, kwargs={"game": game, "config": config,
                                                               "num": i, "opponent": "random"})
                      for i in range(config.services.evaluators)]
    else:
        raise ValueError("As of now, only int is supported for actors")
    try:
        model_broadcaster = dto.ProcessesBroadcaster(actors + evaluators)

        if config.replay_buffer.implementation.type == "grpc":
            table = reverb.Table(
                name=config.replay_buffer.implementation.table,
                sampler=expand_arguments(get_selector(config.replay_buffer.implementation.sampler.name)),
                remover=expand_arguments(get_selector(config.replay_buffer.implementation.remover.name)),
                max_size=config.replay_buffer.buffer_size,
                rate_limiter=reverb.rate_limiters.MinSize(config.replay_buffer.implementation.min_size),
                signature=[tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(1,), dtype=tf.float32),
                           tf.TensorSpec(shape=(), dtype=tf.float32),
                           tf.TensorSpec(shape=(None,), dtype=tf.float32)]
            )
            reverb_server = reverb.Server(tables=[table], port=config.replay_buffer.implementation.port)
            replay_buffer = dto.GrpcReplayBuffer(address=config.replay_buffer.implementation.address, port=config.replay_buffer.implementation.port,
                                                 table=config.replay_buffer.implementation.table)
            actors_orchestrator = dto.ActorsGrpcOrchestrator(actors, server_address=config.replay_buffer.implementation.address,
                                                             server_port=config.replay_buffer.implementation.port, table=config.replay_buffer.implementation.table,
                                                             server_max_workers=10,
                                                             max_buffer_size=config.replay_buffer.buffer_size,
                                                             max_game_length=game.max_game_length(),
                                                             request_length=64)

            thread = threading.Thread(target=dto.ActorsGrpcOrchestrator.collect, args=(actors_orchestrator,),
                                      daemon=True)
            thread.start()

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
        process_broadcaster = dto.ProcessesBroadcaster(actors + evaluators)
        process_broadcaster.request_exit()
        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        # for proc in actors:
        #    while proc.exitcode is None:
        #        while not proc.queue.empty():
        #            proc.queue.get_nowait()
        #        proc.join(JOIN_WAIT_DELAY)
        # for proc in evaluators:
        #    proc.join()
