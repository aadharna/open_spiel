from typing import Sequence

import numpy as np
from open_spiel.python.algorithms.alpha_zero import model_v2 as model_lib
from  mpg.ml.model.gnn import GNN,ResGNN
from mpg.ml.model.mlp import MLP

import tensorflow as tf

from open_spiel.python.algorithms.alpha_zero_mpg.utils import TrainInput

keras = tf.keras

Losses=model_lib.Losses

valid_model_types = ["mlp", "gnn", "residual_mlp","residual_gnn", "res_gnn","res_mlp" ]


class MPGModel(model_lib.ModelV2):

    ADJACENCY_MATRIX_AXIS=0
    WEIGHTS_AXIS=1
    def build(self,config,game):
        env_shape,state_shape = self.config.observation_shape
        num_actions=game.num_distinct_actions()
        graph_size=env_shape[0]
        #TODO: Add support for residual networks
        #TODO: Add support for more model hyper-parameters
        arguments=vars(config.model.arguments)
        match config.nn_model:
            case "mlp":
                model=MLP(graph_size=graph_size,num_actions=num_actions,state_shape=state_shape,name="mlp",
                          **arguments)
            case "gnn":
                model=GNN(name="gnn",**arguments)

            case "residual_mlp" | "res_mlp":
                model=MLP(graph_size=graph_size,num_actions=num_actions,state_shape=state_shape,name="res_mlp",
                          **arguments)
            case "residual_gnn" | "res_gnn":
                model=ResGNN(name="res_gnn",**arguments)
            case _:
                raise ValueError(f"Invalid model type {config.nn_model}")
        return model

    def _get_input(self,batch):
        return {"environment":batch.environment,"state":batch.state}

    def _get_batch(self, train_inputs: Sequence[TrainInput]):
        """Converts a list of TrainInputs to a batch."""
        return TrainInput.stack(train_inputs)

