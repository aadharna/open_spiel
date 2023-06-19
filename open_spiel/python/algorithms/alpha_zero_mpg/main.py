import datetime
import json
import os
import sys
import tempfile

import pyspiel
from open_spiel.python.utils import spawn

JOIN_WAIT_DELAY = 0.001


from .utils import Config
from . import learner, actor, evaluator,utils

class LocalActorFactory:
    def __init__(self,specs):
        self.specs=specs
        pass

    def __call__(self,config,game,queue,num):
        act = actor.MultiProcActor(config, num, name="actor")
        return act.start(queue=queue, game=game)


class LocalEvaluatorFactory:
    def __init__(self, specs):
        self.specs = specs
        pass

    def __call__(self, config, game, queue, num):
        eval = evaluator.MultiProcEvaluator(config, num, name="evaluator")
        return eval.start(queue=queue, game=game)


def create_local_actor(*, game, config, queue, num):
    """Create a local actor process."""
    act= actor.MultiProcActor(config, num, name="actor")
    return act.start(queue=queue, game=game)


def alpha_zero(config: Config):
  """Start all the worker processes for a full alphazero setup."""
  game = pyspiel.load_game(config.game)
  if game.observation_tensor_shape_specs() == pyspiel.TensorShapeSpecs.VECTOR:
      shape=game.observation_tensor_shape()
  else:
      shape=game.observation_tensor_shapes_list()
  config = config._replace(
      observation_shape=shape,
      output_size=game.num_distinct_actions())

  print("Starting game", config.game)
  if game.num_players() != 2:
    sys.exit("AlphaZero can only handle 2-player games.")
  game_type = game.get_type()
  if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    raise ValueError("Game must have terminal rewards.")
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Game must have sequential turns.")
  if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
    raise ValueError("Game must be deterministic.")

  if config.fix_environment:
      print("Fixing environment:")
      print(game.new_initial_state())

  path = config.path
  if not path:
    path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), game.get_type().short_name))
    config = config._replace(path=path)

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.isdir(path):
    sys.exit("{} isn't a directory".format(path))
  print("Writing logs and checkpoints to:", path)
  print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                    config.nn_depth))

  with open(os.path.join(config.path, "config.json"), "w") as fp:
    fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

  actor_factory=LocalActorFactory(config)
  evaluator_factory=LocalEvaluatorFactory(config)
  actors = [spawn.Process(actor_factory, kwargs={"game": game, "config": config,
                                         "num": i})
            for i in range(config.actors)]
  evaluators = [spawn.Process(evaluator_factory, kwargs={"game": game, "config": config,
                                                 "num": i})
                for i in range(config.evaluators)]

  def broadcast(msg):
    for proc in actors + evaluators:
      proc.queue.put(msg)

  try:
    learner.learner(game=game, config=config, actors=actors,  # pylint: disable=missing-kwoa
            evaluators=evaluators, broadcast_fn=broadcast)
  except (KeyboardInterrupt, EOFError):
    print("Caught a KeyboardInterrupt, stopping early.")
  finally:
    broadcast("")
    # for actor processes to join we have to make sure that their q_in is empty,
    # including backed up items
    for proc in actors:
      while proc.exitcode is None:
        while not proc.queue.empty():
          proc.queue.get_nowait()
        proc.join(JOIN_WAIT_DELAY)
    for proc in evaluators:
      proc.join()
