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

from open_spiel.python.algorithms.alpha_zero import model_v2 as model_lib
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("game", None, "Name of the game")
flags.DEFINE_string("path", None, "Directory to save graph")
flags.DEFINE_string("graph_def", None, "Filename for the graph")
flags.DEFINE_enum("nn_model", "resnet", model_lib.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 2 ** 7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 10, "How deep should the network be.")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate used for training")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_bool("verbose", False, "Print information about the model.")
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
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",
        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",

        "quiet",
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



def main(_):
    config = Config(
        game=FLAGS.game,
        path=FLAGS.path,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        checkpoint_freq=FLAGS.checkpoint_freq,
        actors=FLAGS.actors,
        evaluators=FLAGS.evaluators,
        evaluation_window=FLAGS.evaluation_window,
        eval_levels=FLAGS.eval_levels,
        nn_model=FLAGS.nn_model,
        nn_width=FLAGS.nn_width,
        nn_depth=FLAGS.nn_depth,
        observation_shape=None,
        output_size=None,

        quiet=FLAGS.quiet,
    )
    game = pyspiel.load_game(FLAGS.game)
    model = model_lib.ModelV2(config,game)
    model.model.save(os.path.join(config.path,config.graph_def))

    if FLAGS.verbose:
        print("Game:", FLAGS.game)
        print("Model type: %s(%s, %s)" % (FLAGS.nn_model, FLAGS.nn_width,
                                          FLAGS.nn_depth))
        print("Model size:", count_params(model.model.trainable_variables), "variables")
        print("Variables:")
        for v in model.model.trainable_variables:
            print("  ", v.name, v.shape)


if __name__ == "__main__":
    app.run(main)
