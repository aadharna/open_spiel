import abc
import multiprocessing

from . import model as model_lib

class Resource:
    def __init__(self,logger,name=None):
        if name is None:
            name = self.__class__.__name__
        self.name=name
        self._lock = multiprocessing.Lock()
        self.logger=logger

    def update(self,*args,**kwargs):
        self._lock.acquire()
        try:
            self._update(*args,**kwargs)
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
    def _update(self,*args,**kwargs):
        pass


class ModelResource(Resource):
    def __init__(self,logger,config,name=None):
        super().__init__(logger,name)
        self.config = config
        self.model = self._init_model()
        pass

    @abc.abstractmethod
    def _init_model(self):
        pass


class SavedModelBundle(ModelResource):
    def __init__(self,logger,config,game,name=None):
        self.game = game
        super().__init__(logger,config,name)
        self._model = self._init_model()
        pass

    def _init_model(self):
        return model_lib.MPGModel(self.config, self.game)

    def _value(self):
        return self._model

    def _update(self,path:str,az_evaluator):
        logger=self.logger
        if path:
            logger.print("Inference cache:", az_evaluator.cache_info())
            logger.print("Loading checkpoint", path)
            self.model.load_latest_checkpoint()
            az_evaluator.clear_cache()
            return True
        return False