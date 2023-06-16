# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export the model's Tensorflow graph as a protobuf."""
import collections
import os

from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero_mpg import model as model_lib
import pyspiel
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("game", None, "Name of the game")
flags.DEFINE_string("path", None, "Directory to save graph")
flags.DEFINE_string("graph_def", None, "Filename for the graph")
flags.DEFINE_enum("nn_model", "mlp", model_lib.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 2 ** 7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 10, "How deep should the network be.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate used for training")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_float("regularization", 0.0001, "L2 regularization strength.")
flags.DEFINE_bool("verbose", False, "Print information about the model.")
flags.DEFINE_bool("deterministic", False, "Initialize model weights to 0.")
flags.DEFINE_bool("with_checkpoint", False, "Save model checkpoint.")
flags.mark_flag_as_required("game")
flags.mark_flag_as_required("path")
flags.mark_flag_as_required("graph_def")
from keras.utils.layer_utils import count_params


class Config(collections.namedtuple(
    "Config", [
        "game",
        "path",
        "graph_def", # "graph_def" is the name of the file that will be saved
        "learning_rate",
        "weight_decay",
        "regularization",
        "nn_model",
        "nn_width",
        "nn_depth",
        "verbose",
        "observation_shape",
        "output_size",
        "debug",
        "deterministic",
        "with_checkpoint",
    ])):
    """A config for the model/experiment."""
    pass

    @property
    def architecture(self):
        return  self.nn_model

    @property
    def max_actions(self):
        return self.game.max_actions()

    @property
    def num_players(self):
        return self.game.num_players()


class DeterministicRNG:
    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self,shape=None, *args, **kwargs):
        fans=np.prod(shape[:-1]) if shape is not None else 1
        stddev=np.sqrt(2/fans)
        return self.rng.normal(0,stddev,shape)

def main(_):
    config = Config(
        graph_def=FLAGS.graph_def,
        game=FLAGS.game,
        path=FLAGS.path,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        nn_model=FLAGS.nn_model,
        nn_width=FLAGS.nn_width,
        nn_depth=FLAGS.nn_depth,
        regularization=FLAGS.regularization,
        verbose=FLAGS.verbose,
        observation_shape=None,
        output_size=None,
        debug=False,
        deterministic=FLAGS.deterministic,
        with_checkpoint=FLAGS.with_checkpoint
    )
    os.makedirs(config.path,exist_ok=True)
    game = pyspiel.load_game(FLAGS.game)
    if game.observation_tensor_shape_specs() == pyspiel.TensorShapeSpecs.VECTOR:
        shape=game.observation_tensor_shape()
    else:
        shape=game.observation_tensor_shapes_list()
    config = config._replace(
        observation_shape=shape,
        output_size=game.num_distinct_actions())
    model = model_lib.MPGModel(config,game)
    model.model.save(os.path.join(config.path,config.graph_def))

    if config.verbose:
        print("Game:", FLAGS.game)
        print("Model type: %s(%s, %s)" % (FLAGS.nn_model, FLAGS.nn_width,
                                          FLAGS.nn_depth))
        print("Model size:", count_params(model.model.trainable_variables), "variables")
        print("Variables:")
        for v in model.model.trainable_variables:
            print("  ", v.name, v.shape)

    if config.deterministic:
        rng=DeterministicRNG(0)
        model.model.set_weights([rng(w.shape) for w in model.model.get_weights()])
    if config.with_checkpoint:
        os.makedirs(os.path.join(config.path,"checkpoints"),exist_ok=True)
        model.model.save_weights(os.path.join(config.path,"checkpoints",config.graph_def))


if __name__ == "__main__":
    app.run(main)
