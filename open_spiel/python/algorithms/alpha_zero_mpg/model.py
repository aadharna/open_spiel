import collections
import os
from functools import cached_property, cache
from typing import Sequence

import numpy as np
import tensorflow as tf
from cachetools import cached
from open_spiel.python.algorithms.alpha_zero import model_v2 as model_lib
import tensorflow as tf
keras = tf.keras

Losses=model_lib.Losses

valid_model_types = ["mlp", "conv2d", "resnet" ]

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
        self.input_environment = keras.layers.Input(shape=env_shape,
                                                    name="environment")  # s: batch_size x board_x x board_y
        self.input_state = keras.layers.Input(shape=state_shape, name="state")
        state_reshape = keras.layers.Reshape((1,))(self.input_state)
        state_encoding=keras.layers.CategoryEncoding(num_tokens=game.num_distinct_actions(),output_mode="one_hot",name="state_encoding")(state_reshape)
        flattened = keras.layers.Flatten(name="flatten")(self.input_environment)
        stack = keras.layers.Concatenate()([flattened, state_encoding])

        legals_mask = keras.layers.Dot(axes=(-2,-1), name="legals_mask")([self.input_environment[...,self.ADJACENCY_MATRIX_AXIS], state_encoding])

        y = keras.layers.BatchNormalization()(stack)
        y = keras.layers.Dense(128)(y)
        z = keras.layers.BatchNormalization()(y)
        self.pi = keras.layers.Dense(self.action_size, activation="softmax", name="policy_targets_unmasked")(
            z)  # batch_size x self.action_size
        self.pi = keras.layers.Multiply(name="policy_targets")([self.pi, legals_mask])
        self.v = keras.layers.Dense(1, activation="tanh", name="value_targets")(z)  # batch_size x 1
        model = keras.models.Model(inputs=[self.input_environment, self.input_state],
                                   outputs=[self.v, self.pi])
        return model

    def _get_input(self,batch):
        return {"environment":batch.environment,"state":batch.state}

    def _get_batch(self, train_inputs: Sequence[TrainInput]):
        """Converts a list of TrainInputs to a batch."""
        return TrainInput.stack(train_inputs)

