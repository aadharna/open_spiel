import abc
from typing import List, Callable

import numpy as np
from scipy.special import softmax
from .utils import FloatArrayLikeType

class Guide(abc.ABC):

    @abc.abstractmethod
    def call(self, priors: FloatArrayLikeType, payoffs: FloatArrayLikeType) -> FloatArrayLikeType:
        pass

    def __call__(self, priors, payoffs):
        return self.call(priors, payoffs)


class SoftmaxGuide(Guide):

    def call(self, priors: FloatArrayLikeType, payoffs: FloatArrayLikeType):
        priors = np.array(priors)
        payoffs = np.array(payoffs)
        measure = priors * softmax(payoffs)
        return measure / np.mean(measure)


class PayoffsPrior:

    @abc.abstractmethod
    def call(self, payoffs: FloatArrayLikeType):
        pass

    def __call__(self, payoffs: FloatArrayLikeType):
        return self.call(payoffs)


class PayoffsSoftmaxPrior(PayoffsPrior):

    def __init__(self, scale=1):
        self.scale = scale

    def call(self, payoffs: FloatArrayLikeType):
        payoffs = np.array(payoffs)
        return softmax(self.scale * payoffs)


class PayoffsReLUPrior(PayoffsPrior):

    def __init__(self, shift=0):
        self.shift = shift
        pass

    def call(self, payoffs: FloatArrayLikeType):
        payoffs = np.array(payoffs)
        guide = np.array(payoffs + self.shift >= 0, payoffs, 0)
        if np.all(guide == 0):
            return payoffs
        return guide / np.mean(guide)


class PayoffsPowerPrior(PayoffsPrior):

    def __init__(self, k=1, r=1):
        self.k = k
        self.r = r

    def call(self, payoffs: FloatArrayLikeType):
        payoffs = np.array(payoffs)
        abs_payoffs = np.abs(payoffs)
        K = np.power(abs_payoffs, self.k)
        guide = np.where(payoffs > 0, K + self.r, 1. / (K + self.r))
        return guide / np.mean(guide)


class LinearGuide(Guide):
    def __init__(self, t, payoffs_prior):
        if t < 0 or t > 1:
            raise ValueError(f"t must be in the interval [0,1]. Got {t}")
        super().__init__()
        self.t = t
        self.payoffs_prior = payoffs_prior

    def call(self, priors: List[float], payoffs: FloatArrayLikeType):
        priors = np.array(priors)
        payoffs = np.array(payoffs)
        t = self.t
        return t * priors + (1 - t) * self.payoffs_prior(payoffs)
