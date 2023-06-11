import collections
import os
from functools import cached_property, cache
from typing import Sequence

import numpy as np
import tensorflow as tf
from cachetools import cached




keras = tf.keras

valid_model_types = ["mlp", "conv2d", "resnet" , "mpgnet"]


class L2Loss(tf.Module):
    def __init__(self, model, alpha:float=1, name="l2_loss"):
        super().__init__(name=name)
        self.model = model
        self.alpha=alpha

    @tf.function
    def __call__(self):
        return self.alpha*tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
def l2_loss(model,alpha:float=1):
    @tf.function
    def l2_loss_implementation():
        return alpha*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
    return l2_loss_implementation

class L2LossHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, model,regularization=0.01):
        super().__init__()
        self.model = model
        self.l2_loss=L2Loss(self.model,alpha=regularization, name="l2_loss_callback")

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
        self.checkpoint = None
        self.config = config
        self.action_size=game.num_distinct_actions()
        regularization=config.regularization

        # Neural Net
        self.model=self.build(config,game)
        self.regularization=regularization
        self.l2_loss_function=L2Loss(self.model,alpha=regularization, name="l2_loss")
        #self.model.add_loss(l2_loss(self.model,alpha=regularization))
        self.model.add_loss(self.l2_loss_function)
        self.model.compile(loss={"policy_targets":"categorical_crossentropy", "value_targets":"mean_squared_error"},
                           optimizer=keras.optimizers.Adam(config.learning_rate))


    def build(self,config,game) -> tf.keras.Model:
        input_shape = self.config.observation_shape
        match self.config.architecture:
            case "mlp":
                self.input_state = keras.layers.Input(shape=input_shape,
                                                      name="input")  # s: batch_size x board_x x board_y
                self.input_mask=keras.layers.Input(shape=(self.action_size,),name="legals_mask")
                flattened = keras.layers.Flatten(name="flatten")(self.input_state)
                y = keras.layers.BatchNormalization()(flattened)
                y = keras.layers.Dense(128, activation="relu")(y)
                z = keras.layers.BatchNormalization()(y)
                self.pi = keras.layers.Dense(self.action_size, activation="softmax", name="policy_targets_unmasked")(z)  # batch_size x self.action_size
                self.pi = keras.layers.Multiply(name="policy_targets")([self.pi,self.input_mask])
                self.v = keras.layers.Dense(1, activation="tanh", name="value_targets")(z)  # batch_size x 1
                model = keras.Model(inputs=[self.input_state,self.input_mask], outputs=[self.v,self.pi])
            case _:
                raise ValueError(f"Invalid architecture {self.config.architecture}")
        return model
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

    @cache
    def count_trainable_variables(self):
        return self.model.count_params()

    def save_checkpoint(self, path: str):
        """
    Saves the model to a checkpoint
    :param path: path to the checkpoint
    """
        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(self.model)
        return self.checkpoint.save(path)

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

    def load_latest_checkpoint(self):
        """
        Loads the latest checkpoint
        :return:
        """
        return self.load_checkpoint(tf.train.latest_checkpoint(os.path.join(self.config.path, "checkpoints")))

    def save_checkpoint_counter(self, counter):
        """
    Saves the model to a checkpoint
    :param counter: the counter
    """
        return self.save_checkpoint(os.path.join(self.config.path, "checkpoints", f"model_{counter}"))

    def predict(self, *inputs):
        """
    :param env: the environment
    :param state: the state
    :return: pi, v
    """
        return self.model.predict(inputs,verbose = 0)

    def loss_function(self):
        return self.model.loss

    def inference(self, *inputs):
        return self.predict(*inputs)

    def _get_batch(self, train_inputs: Sequence[TrainInput]):
        """Converts a list of TrainInputs to a batch."""
        return TrainInput.stack(train_inputs)

    def _get_input(self,batch):
        return {"input":batch.observation,"legals_mask":batch.legals_mask}

    def _get_output(self,batch):
        return {"policy_targets":batch.policy, "value_targets":batch.value}

    def update(self, train_inputs: Sequence[TrainInput]):
        """Runs a training step."""
        batch = self._get_batch(train_inputs)
#        print(batch.observation.shape)

        # Run a training step and get the losses.
        x=self._get_input(batch)
        y=self._get_output(batch)
        log=self.model.fit(x, y, batch_size=self.config.train_batch_size, epochs=3, verbose=1,
                           callbacks=[L2LossHistoryCallback(self.model,self.regularization),keras.callbacks.CSVLogger(self.config.path+"/log.csv",append=True)])
        return Losses(np.mean(log.history["policy_targets_loss"]), np.mean(log.history["value_targets_loss"]),
                      np.mean(log.history["l2_loss"]))