import abc
import multiprocessing

import numpy as np

from . import model as model_lib, utils


class Resource:
    def __init__(self, logger, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self._lock = multiprocessing.Lock()
        self.logger = logger

    def update(self, *args, **kwargs):
        self._lock.acquire()
        try:
            self._update(*args, **kwargs)
            return True
        finally:
            self._lock.release()
        pass

    @property
    def value(self):
        self._lock.acquire()
        try:
            return self._value()
        finally:
            self._lock.release()
        pass

    @abc.abstractmethod
    def _value(self):
        pass

    @abc.abstractmethod
    def _update(self, *args, **kwargs):
        pass


class ModelResource(Resource):
    def __init__(self, logger, config, name=None):
        super().__init__(logger, name)
        #We fix the seed here so that the hashing function is deterministic
        self.config = config
        self._model = self._init_model()
        # This is does not need to be a strong hash function, just needs to check if the model has changed
        self.hasher = utils.AlmostUniversalHasher.deterministic_instance()
        pass


    @abc.abstractmethod
    def _init_model(self):
        pass

    @classmethod
    def from_model(cls, model,logger,config,name=None):
        resource =  cls.__new__(cls)
        resource._lock = multiprocessing.Lock()
        resource.model = model
        resource.config = config
        resource.logger = logger
        resource.name = name
        resource.hasher = utils.AlmostUniversalHasher.deterministic_instance()

        return resource

    def _value(self):
        return self._model

    def hash(self):
        variables = self._model.trainable_variables
        hash_list = []
        # We hope that the list is guaranteed to be in the same order
        for var in variables:
            hash_list.append(int.from_bytes(var.numpy().tobytes(), byteorder="big"))
        return self.hasher.hash(hash_list)

class SavedModelBundle(ModelResource):
    def __init__(self, logger, config, game, name=None):
        self.game = game
        super().__init__(logger, config, name)
        pass

    def _init_model(self):
        return model_lib.MPGModel(self.config, self.game)

    def _update(self, path: str, az_evaluator):
        logger = self.logger
        if path:
            logger.print("Inference cache:", az_evaluator.cache_info())
            logger.print("Loading checkpoint", path)
            self._model.load_latest_checkpoint()
            az_evaluator.clear_cache()
            return True
        return False
