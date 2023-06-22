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

"""Starting point for playing with the AlphaZero algorithm."""
import os
import sys

import yaml
from absl import app
from absl import flags
from argparse import Namespace

from open_spiel.python.algorithms.alpha_zero_mpg import alpha_zero as alpha_zero_v1
from open_spiel.python.algorithms.alpha_zero_mpg import main as alpha_zero_v2
from open_spiel.python.algorithms.alpha_zero_mpg import model as model_lib
from open_spiel.python.algorithms.alpha_zero_mpg import utils
from open_spiel.python.utils import spawn

def nested_dict_to_namespace(nested_dict):
    """Converts a nested dict to a namespace."""
    if type(nested_dict) == Namespace:
        return nested_dict_to_namespace(vars(nested_dict))
    if type(nested_dict) != dict and type(nested_dict) != list:
        return nested_dict
    namespace = Namespace()
    if type(nested_dict) is dict:
        for key, value in nested_dict.items():
            setattr(namespace, key, nested_dict_to_namespace(value))

    elif type(nested_dict) is list:
        namespace = [None]*len(nested_dict)
        for i, value in enumerate(nested_dict):
            namespace[i]=nested_dict_to_namespace(value)
    return namespace

def compatibility_mode(config):
    config.fix_environment = config.game.fix_environment
    config.temperature = config.mcts.temperature
    config.temperature_drop = config.mcts.temperature_drop
    config.mcts.policy_epsilon = config.mcts.policy_epsilon
    config.mcts.policy_alpha = config.mcts.policy_alpha
    config.mcts.max_simulations = config.mcts.max_simulations
    config.mcts.temperature_drop = config.mcts.temperature_drop
    config.max_moves = config.game.max_moves
    config.nn_model = config.model.architecture
    config.nn_width = config.model.width
    config.nn_depth = config.model.depth
    config.training_batch_size = config.training.batch_size
    config.learning_rate = config.training.learning_rate
    config.weight_decay = config.training.weight_decay
    config.checkpoint_freq = config.training.checkpoint_freq
    config.max_steps = config.training.max_steps
    config.steps_per_epoch = config.training.steps_per_epoch
    config.epochs_per_iteration = config.training.epochs_per_iteration
    config.evaluation_window = config.services.evaluation_window
    config.regularization = config.training.weight_decay
    config.policy_epsilon = config.mcts.policy_epsilon
    config.policy_alpha = config.mcts.policy_alpha
    config.uct_c = config.mcts.uct_c
    config.replay_buffer_size = config.replay_buffer.buffer_size
    config.replay_buffer_reuse = config.replay_buffer.reuse
    config.max_simulations = config.mcts.max_simulations
    config.train_batch_size = config.training.batch_size



def main(unused_argv):
    try:
        path=os.path.dirname(__file__)
        file="".join(os.path.basename(__file__).split(".")[:-1])+".yml"
        with open(os.path.join(path,file), "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            config = nested_dict_to_namespace(config)
    except FileNotFoundError:
        print("No config file found. Using default config.", file=sys.stderr)
    # For compatibility with the old config file
    compatibility_mode(config)
    if config.specification == 1:
        alpha_zero_v1.alpha_zero(config)
    elif config.specification == 2:
        alpha_zero_v2.alpha_zero(config)
    else:
        raise ValueError("Invalid version: {}".format(config.version))


if __name__ == "__main__":
    with spawn.main_handler():
        main(sys.argv)
