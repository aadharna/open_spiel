import os
import re
import numpy as np
import pickle as pkl
from evaluator import AlphaZeroEvaluator
from alpha_zero import _init_model_from_config
from collections import OrderedDict

from open_spiel.python.algorithms import mcts


class AZPopulationWithEvaluators(mcts.Evaluator):
  def __init__(self, game, init_bot_fn, model, config, k=5):
    self.game = game
    self.config = config
    # each individual of the population gets 1/5 of the max simulation budget
    self.config = config._replace(max_simulations=config.max_simulations // 5)
    self.model = model
    self.k = k
    self.threshold = 0.15
    self.init_bot_fn = init_bot_fn

    self.current_agent = init_bot_fn(config, game, AlphaZeroEvaluator(self.game, self.model), False)
    
    init_hist_model = _init_model_from_config(self.config)
    init_hist_chkpt = init_bot_fn(config, game, AlphaZeroEvaluator(self.game, init_hist_model), False)
    self.checkpoint_evaluators = {'checkpoint-0': init_hist_chkpt.evaluator}
    self.checkpoint_mcts_bots = {'checkpoint-0': init_hist_chkpt}


    init_nov_model = _init_model_from_config(self.config)
    init_nov_chkpt = init_bot_fn(config, game, AlphaZeroEvaluator(self.game, init_nov_model), False)
    self.novelty_evaluators = {'checkpoint-0': init_nov_chkpt.evaluator}
    self.novelty_mcts_bots = {'checkpoint-0': init_nov_chkpt}
    
    self.A = np.array([[0]])
  
  def add_checkpoint_bot(self, checkpoint_path):
    model = _init_model_from_config(self.config)
    model.load_checkpoint(checkpoint_path)
    self.checkpoint_evaluators[checkpoint_path] = AlphaZeroEvaluator(self.game, model)
    self.checkpoint_mcts_bots[checkpoint_path] = self.init_bot_fn(self.config, self.game, self.checkpoint_evaluators[checkpoint_path], False)


  def add_novelty_bot(self, checkpoint_path):
    model = _init_model_from_config(self.config)
    model.load_checkpoint(checkpoint_path)
    self.novelty_evaluators[checkpoint_path] = AlphaZeroEvaluator(self.game, model)
    self.novelty_mcts_bots[checkpoint_path] = self.init_bot_fn(self.config, self.game, self.novelty_evaluators[checkpoint_path], False)

  def is_novel(self, a):
    # A is a matrix of outcomes
    # a is a vector of outcomes
    # return True if a is novel
    # calculate the knn distance between a and a's knn in A
    # if that distance is greater than some threshold, return True, dist
    # else return False, dist
    a = np.array(a)
    if self.A.shape[1] != a.shape[0]:
      print(self.A, a)
      raise ValueError('Novelty archive and response vector do not match in dims')
    dists = np.linalg.norm(self.A - a, axis=1)
    knn = np.argsort(dists)[:self.k]
    knn_dists = dists[knn]
    avg_dist = np.mean(knn_dists)
    normalized_dist = avg_dist / (2 * np.sqrt(a.shape[0]))
    return normalized_dist >= self.threshold, normalized_dist

  
  def evaluate(self, state):
    working_state = state.clone()
    # play a rollout against each checkpoint bot

    # check if the number of historical bots is equal to the number of columns in A
    if self.A.shape[1] != len(self.checkpoint_mcts_bots):
      # load all missing checkpoint bots
      all_checkpoints = [f.split('.')[0] for f in 
                            os.listdir(self.model._path) if
                            re.match(r'.*historical.*', f)]
      checkpoint_paths = list(set(checkpoint_paths))
      for f in all_checkpoints:
          full_path = os.path.join(self.model._path, f)
          if full_path not in self.checkpoint_mcts_bots:
              self.add_checkpoint_bot(full_path)


    checkpoint_results = []
    for evaluator_idx, bot in self.checkpoint_mcts_bots.items():
      p1, p2 = self.guided_rollout(working_state, 
                                   self.current_agent, 
                                   bot)
      checkpoint_results.append(p2)
      # p1, p2 = bot.evaluator.evaluate(working_state) # how well do I, the opponent, think I'll do in this board state?
      # checkpoint_results.append(p1)

    if len(checkpoint_results) > 1:
      # check novelty of response vector
      is_novel, dist = self.is_novel(checkpoint_results)
    else:
      dist, _ = self.current_agent.evaluator.evaluate(working_state) 
    return [dist, -dist]
  
  def prior(self, state):
    return self.current_agent.evaluator.prior(state)

  def guided_rollout(self, state, p1_bot, p2_bot, n=1):
    results = [0, 0]
    for _ in range(n):
      working_state = state.clone()
      bots = [p1_bot, p2_bot]

      while not working_state.is_terminal():
        # if it is p2's turn, the state current player will tell us so we don't need to change anything
        current_player = working_state.current_player()
        bot = bots[current_player]
        actions_and_probs = bot.evaluator.prior(working_state)
        actions = [a[0] for a in actions_and_probs]
        probs = [a[1] for a in actions_and_probs]
        # take the softmax of the probs
        probs = np.exp(probs) / np.sum(np.exp(probs))
        action = np.random.choice(actions, p=probs)
        working_state.apply_action(action)
      result = working_state.returns()
      p1_result = result[0]
      p2_result = result[1]
      results[0] += p1_result
      results[1] += p2_result

    p1_result = results[0] / n
    p2_result = results[1] / n
    return p1_result, p2_result
  
  def update_current_agent_to_main(self, checkpoint_path):
    self.current_agent.evaluator._model.load_checkpoint(checkpoint_path)
    self.current_agent.evaluator.clear_cache()

  def update_response_matrix(self, novelty_archive_path):
    # load the novelty archive
    self.A = np.load(novelty_archive_path)



if __name__ == '__main__':
  import pyspiel

  evaluators = [mcts.RandomRolloutEvaluator(n_rollouts=5) for _ in range(2)]

  game = pyspiel.load_game("connect_four")
  random_state = np.random.RandomState(42)
  uct_c = 2
  child_selection_fn = mcts.SearchNode.puct_value

  state = game.new_initial_state()

  values = evaluators[0].evaluate(state)
  prior = evaluators[0].prior(state)


  bot = mcts.MCTSBot(game, uct_c, 25, evaluators[0], solve=False, random_state=random_state,
                child_selection_fn=child_selection_fn, dirichlet_noise=None, verbose=False, dont_return_chance_node=False)
  
  state = game.new_initial_state()
  action = bot.step(state)

  novelty_bot = ...
  