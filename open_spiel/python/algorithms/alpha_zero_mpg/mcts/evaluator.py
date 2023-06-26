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

"""An MCTS Evaluator for an AlphaZero model."""
import enum
from typing import List, Callable, Union, Iterable

import numpy as np

from open_spiel.python.algorithms import mcts
import pyspiel
from open_spiel.python.utils import lru_cache
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import tensorflow as tf
from open_spiel.python.algorithms.alpha_zero_mpg.utils import nested_reshape
from .utils import FloatArrayLikeType

from .. import resource


class StateRepresentation(enum.Enum):
    """State representation for AlphaZero."""
    COMPRESSED = "compressed"
    MINIMAL = "minimal"
    NORMAL = "normal"
    HASH = "hash"


class AlphaZeroEvaluator(evaluator_lib.AlphaZeroEvaluator):
    """An AlphaZero MCTS Evaluator."""

    def __init__(self, game, model, cache_size=2 ** 16,*, state_representation=StateRepresentation.NORMAL):
        if isinstance(model, resource.SavedModelBundle):
            self._model_bundle=model
            model=model.value
        else:
            self._model_bundle=None
        super().__init__(game, model, cache_size)
        self.state_representation = StateRepresentation(state_representation)

    def cache_info(self):
        return self._cache.info()

    def clear_cache(self):
        self._cache.clear()

    def _get_cache_key(self,game_state, environment_tensor, state_tensor):
        if self.state_representation == StateRepresentation.COMPRESSED:
            raise NotImplementedError("Compressed state representation not implemented")
        elif self.state_representation == StateRepresentation.MINIMAL:
            return (game_state.current_player(), int(state_tensor))
        elif self.state_representation == StateRepresentation.NORMAL:
            return (game_state.current_player(), environment_tensor.tobytes(), state_tensor.tobytes())
        elif self.state_representation == StateRepresentation.HASH:
            return (game_state.current_player(),game_state.graph_size(), hash(state_tensor))
        else:
            raise ValueError("Unknown state representation: {}".format(self.state_representation))

    def _inference(self, game_state):
        # Make a singleton batch
        environment, state = nested_reshape(game_state.observation_tensor(), game_state.observation_tensor_shapes_list())
        environment = np.expand_dims(environment, 0)
        state = np.expand_dims(state, 0)

        cache_key = self._get_cache_key(game_state, environment, state)
        value, policy = self._cache.make(
            cache_key, lambda: self.model.inference(environment, state))
        return value[0, 0], policy[0]  # Unpack batch

    def evaluate(self, state):
        """Returns a value for the given state."""
        value, _ = self._inference(state)
        return np.array([value, -value])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            # Returns the probabilities for all actions.
            _, policy = self._inference(state)
            return [(action, policy[action]) for action in state.legal_actions()]

    @property
    def model(self):
        if self._model_bundle is not None:
            return self._model_bundle.value
        else:
            return self._model


class GuidedAlphaZeroEvaluator(AlphaZeroEvaluator):
    """
    an Alpha Zero evaluator with a guide function.
    A guide function is a function that takes the prior probabilities and the action payoffs and gives a more informed guess
    of the transition probabilities
    """
    def __init__(self, game, model, policy_map: Callable[[FloatArrayLikeType,FloatArrayLikeType],FloatArrayLikeType], cache_size=2 ** 16):
        super().__init__(game,model,cache_size)
        self.policy_map=policy_map

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            payoffs=state.legal_actions_with_payoffs()
            legal_actions=payoffs.keys()
            _,model_prior=self._inference(state)
            payoffs_numpy = np.zeros_like(model_prior)
            for u in payoffs:
                payoffs_numpy[u]=payoffs[u]
            policy=self.policy_map(model_prior,payoffs_numpy)
            return [(action, policy[action]) for action in legal_actions]


class StochasticEvaluator(mcts.Evaluator):
    def __init__(self, evaluators:List[mcts.Evaluator], weights:List[float]=None, seed=None):
        self.evaluators = evaluators
        if weights is None:
            weights = np.ones(len(evaluators))
        weights /= np.sum(weights)
        self.weights = weights
        self.rng = np.random.RandomState(seed)

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            # Returns the probabilities for all actions.
            _, policy = self.evaluate(state)
            return [(action, policy[action]) for action in state.legal_actions()]

    def evaluate(self, state):
        """Returns a value for the given state."""
        return self.rng.choice(self.evaluators, p=self.weights).evaluate(state)


RandomRolloutEvaluator = mcts.RandomRolloutEvaluator