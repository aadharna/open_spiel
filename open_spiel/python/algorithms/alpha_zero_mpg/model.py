import collections
import os
from functools import cached_property, cache
from typing import Sequence

import numpy as np
import tensorflow as tf
from cachetools import cached
from open_spiel.python.algorithms.alpha_zero import model_v2 as model_lib
from  mpg.ml.model.gnn import GNN
from mpg.ml.model.mlp import MLP

import tensorflow as tf
keras = tf.keras

Losses=model_lib.Losses

valid_model_types = ["mlp", "gnn", "residual_mlp","residual_gnn" ]

def nested_reshape(flat_array, shapes_list):
    arrays=[]
    start=0
    for shape in shapes_list:
        size=np.prod(shape)
        arrays.append(np.reshape(flat_array[start:start+size],shape))
        start+=size
    if start!=len(flat_array):
        raise ValueError("Shapes don't match")
    return arrays

class TrainInput(collections.namedtuple(
    "TrainInput", ["environment", "state", "policy", "value"])):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs):
        environment, state, policy, value = zip(*train_inputs)
        return TrainInput(
            np.array(environment, dtype=np.float32),
            np.array(state, dtype=np.int32),
            np.array(policy),
            np.expand_dims(value, 1)
        )
class MPGModel(model_lib.ModelV2):

    ADJACENCY_MATRIX_AXIS=0
    WEIGHTS_AXIS=1
    def build(self,config,game):
        env_shape,state_shape = self.config.observation_shape
        num_actions=game.num_distinct_actions()
        graph_size=env_shape[0]
        #TODO: Add support for residual networks
        #TODO: Add support for more model hyper-parameters
        match config.nn_model:
            case "mlp":
                model=MLP(graph_size=graph_size,num_actions=num_actions,state_shape=state_shape,name="mlp")
            case "gnn":
                model=GNN(name="gnn")
            case _:
                raise ValueError(f"Invalid model type {config.nn_model}")
        return model

    def _get_input(self,batch):
        return {"environment":batch.environment,"state":batch.state}

    def _get_batch(self, train_inputs: Sequence[TrainInput]):
        """Converts a list of TrainInputs to a batch."""
        return TrainInput.stack(train_inputs)

