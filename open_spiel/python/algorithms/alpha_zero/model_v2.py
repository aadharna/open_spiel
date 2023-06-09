import collections
import os
from functools import cached_property
from typing import Sequence

import numpy as np
import tensorflow as tf
from cachetools import cached




keras = tf.keras

valid_model_types = ["mlp", "conv2d", "resnet" , "mpgnet"]

def l2_loss(model,alpha=1):
    return lambda :alpha*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])

class L2LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.l2_loss=l2_loss(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["l2_loss"] = float(self.l2_loss())



class TrainInput(collections.namedtuple(
    "TrainInput", "observation legals_mask policy value")):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs):
        observation, legals_mask, policy, value = zip(*train_inputs)
        return TrainInput(
            np.array(observation, dtype=np.float32),
            np.array(legals_mask, dtype=bool),
            np.array(policy),
            np.expand_dims(value, 1))


class Losses(collections.namedtuple("Losses", "policy value l2")):
    """Losses from a training step."""

    @property
    def total(self):
        return self.policy + self.value + self.l2

    def __str__(self):
        return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
                "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

    def __add__(self, other):
        return Losses(self.policy + other.policy,
                      self.value + other.value,
                      self.l2 + other.l2)

    def __truediv__(self, n):
        return Losses(self.policy / n, self.value / n, self.l2 / n)


class ModelV2:
    """The model."""

    def __init__(self, config,game):
        # game params
        self.config = config
        input_shape = config.observation_shape
        self.action_size=game.num_distinct_actions()
        architecture = config.architecture

        # Neural Net
        match architecture:
            case "mlp":
                self.input_state = keras.layers.Input(shape=input_shape,
                                                      name="environment")  # s: batch_size x board_x x board_y
                flattened = keras.layers.Flatten(name="flatten")(self.input_state)
                y = keras.layers.BatchNormalization()(flattened)
                y = keras.layers.Dense(128, activation="relu")(y)
                z = keras.layers.BatchNormalization(beta_regularizer=keras.regularizers.l2(1),
                                                    gamma_regularizer=keras.regularizers.l2(1))(y)
                self.pi = keras.layers.Dense(self.action_size, activation="softmax", name="pi")(
                    z)  # batch_size x self.action_size
                self.v = keras.layers.Dense(1, activation="tanh", name="v")(z)  # batch_size x 1
                self.model = keras.Model(inputs=self.input_state, outputs=[self.pi, self.v])
            case "mpgnet":
                self.input_environment = keras.layers.Input(shape=input_shape,
                                                            name="environment")  # s: batch_size x board_x x board_y
                self.input_state = keras.layers.Input(shape=(), name="state")
                state_reshape = keras.layers.Reshape((1,))(self.input_state)
                flattened = keras.layers.Flatten(name="flatten")(self.input_environment)
                stack = keras.layers.Concatenate()([flattened, state_reshape])
                y = keras.layers.BatchNormalization()(stack)
                y = keras.layers.Dense(128)(y)
                z = keras.layers.BatchNormalization()(y)
                self.pi = keras.layers.Dense(self.action_size, activation="softmax", name="pi")(
                    z)  # batch_size x self.action_size
                self.v = keras.layers.Dense(1, activation="tanh", name="v")(z)  # batch_size x 1
                self.model = keras.models.Model(inputs=[self.input_environment, self.input_state],
                                                outputs=[self.pi, self.v])
            case _:
                raise ValueError(f"Invalid architecture {architecture}")
        self.model.add_loss(l2_loss(self.model,alpha=1))
        self.model.compile(loss={"pi":"categorical_crossentropy", "v":"mean_squared_error"},
                           optimizer=keras.optimizers.Adam(config.learning_rate))

    @classmethod
    def from_checkpoint(cls, path: str,config):
        """
      Builds a graph from a checkpoint
      :param path: path to the checkpoint
      :return: ModelV2
      """
        instance = ModelV2.__new__(cls)
        instance.model = keras.models.load_model(path)
        instance.input_environment = instance.model.get_layer("environment").input
        instance.input_state = instance.model.get_layer("state").input
        instance.pi = instance.model.get_layer("pi").output
        instance.v = instance.model.get_layer("v").output
        instance.config=config
        return instance

    @cached
    def count_trainable_variables(self):
        return self.model.count_params()

    def save_checkpoint(self, path: str):
        """
    Saves the model to a checkpoint
    :param path: path to the checkpoint
    """
        checkpoint = tf.train.Checkpoint(self.model)
        checkpoint.save(path)
        return path

    def load_checkpoint(self, path: str):
        """
        Loads the model from a checkpoint
        :param path:
        :return:
        """
        checkpoint = tf.train.Checkpoint(self.model)
        # Restore the checkpointed values to the `model` object.
        checkpoint.restore(path)
        return path


    def save_checkpoint_counter(self, counter):
        """
    Saves the model to a checkpoint
    :param counter: the counter
    """
        return self.save_checkpoint(os.path.join(self.config.path, "checkpoints", f"model_{counter}"))

    def predict(self, env, state):
        """
    :param env: the environment
    :param state: the state
    :return: pi, v
    """
        return self.model.predict([env, state])

    def loss_function(self):
        return self.model.loss

    def inference(self, env, state):
        return self.predict(env, state)
    def update(self, train_inputs: Sequence[TrainInput]):
        """Runs a training step."""
        batch = TrainInput.stack(train_inputs)

        # Run a training step and get the losses.
        log=self.model.fit(batch.observation, {"pi":batch.policy, "v":batch.value}, batch_size=self.config.batch_size, epochs=1, verbose=0,
                           callbacks=[L2LossHistoryCallback(self.model),keras.callbacks.CSVLogger(self.config.path+"/log.csv",append=True)])
        return Losses(log.history["pi_loss"][0], log.history["v_loss"][0], log.history["l2_loss"][0])