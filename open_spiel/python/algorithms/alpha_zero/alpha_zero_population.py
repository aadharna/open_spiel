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

"""A basic AlphaZero implementation.

This implements the AlphaZero training algorithm. It spawns N actors which feed
trajectories into a replay buffer which are consumed by a learner. The learner
generates new weights, saves a checkpoint, and tells the actors to update. There
are also M evaluators running games continuously against a standard MCTS+Solver,
though each at a different difficulty (ie number of simulations for MCTS).

Due to the multi-process nature of this algorithm the logs are written to files,
one per process. The learner logs are also output to stdout. The checkpoints are
also written to the same directory.

Links to relevant articles/papers:
  https://deepmind.com/blog/article/alphago-zero-starting-scratch has an open
    access link to the AlphaGo Zero nature paper.
  https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
    has an open access link to the AlphaZero science paper.
"""

import collections
import datetime
import functools
import itertools
import json
import os
import re
import random
import sys
import tempfile
import time
import traceback

import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
import pyspiel
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import file_logger
from open_spiel.python.utils import spawn
from open_spiel.python.utils import stats

import model as model_lib
from population_novelty_evaluator import AZPopulationWithEvaluators

# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


class TrajectoryState(object):
    """A particular point along a trajectory."""

    def __init__(self, observation, current_player, legals_mask, action, policy,
                 value, opponent_id=-1):
        self.observation = observation
        self.current_player = current_player
        self.legals_mask = legals_mask
        self.action = action
        self.policy = policy
        self.value = value
        self.opponent_id = opponent_id


class Trajectory(object):
    """A sequence of observations, actions and policies, and the outcomes."""

    def __init__(self):
        self.states = []
        self.returns = None
        self.novely = None

    def add(self, information_state, action, policy):
        self.states.append((information_state, action, policy))


class Buffer(object):
    """A fixed size buffer that keeps the newest values."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []
        self.total_seen = 0  # The number of items that have passed through.

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    def append(self, val):
        return self.extend([val])

    def extend(self, batch):
        batch = list(batch)
        self.total_seen += len(batch)
        self.data.extend(batch)
        self.data[:-self.max_size] = []

    def sample(self, count):
        return random.sample(self.data, count)


class Config(collections.namedtuple(
    "Config", [
        "game",
        "path",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "replay_buffer_size",
        "replay_buffer_reuse",
        "max_steps",
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",

        "uct_c",
        "max_simulations",
        "policy_alpha",
        "policy_epsilon",
        "temperature",
        "temperature_drop",

        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",

        "quiet",
        "novelty",
    ])):
    """A config for the model/experiment."""
    pass


def _init_model_from_config(config):
    return model_lib.Model.build_model(
        config.nn_model,
        config.observation_shape,
        config.output_size,
        config.nn_width,
        config.nn_depth,
        config.weight_decay,
        config.learning_rate,
        config.path)


def watcher(fn):
    """A decorator to fn/processes that gives a logger and logs exceptions."""

    @functools.wraps(fn)
    def _watcher(*, config, num=None, **kwargs):
        """Wrap the decorated function."""
        name = fn.__name__
        if num is not None:
            name += "-" + str(num)
        with file_logger.FileLogger(config.path, name, config.quiet) as logger:
            print("{} started".format(name))
            logger.print("{} started".format(name))
            try:
                return fn(config=config, logger=logger, **kwargs)
            except Exception as e:
                logger.print("\n".join([
                    "",
                    " Exception caught ".center(60, "="),
                    traceback.format_exc(),
                    "=" * 60,
                ]))
                print("Exception caught in {}: {}".format(name, e))
                raise
            finally:
                logger.print("{} exiting".format(name))
                print("{} exiting".format(name))

    return _watcher


def _init_bot(config, game, evaluator_, evaluation):
    """Initializes a bot."""
    noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
    return mcts.MCTSBot(
        game,
        config.uct_c,
        config.max_simulations,
        evaluator_,
        solve=False,
        dirichlet_noise=noise,
        child_selection_fn=mcts.SearchNode.puct_value,
        verbose=False,
        dont_return_chance_node=True)


def _play_game(logger, game_num, game, bots, temperature, temperature_drop):
    """Play one game, return the trajectory."""
    trajectory = Trajectory()
    actions = []
    state = game.new_initial_state()
    random_state = np.random.RandomState()
    logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
    logger.opt_print("Initial state:\n{}".format(state))

    while not state.is_terminal():
        if state.is_chance_node():
            # For chance nodes, rollout according to chance node's probability
            # distribution
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random_state.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            root = bots[state.current_player()].mcts_search(state)
            policy = np.zeros(game.num_distinct_actions())
            for c in root.children:
                policy[c.action] = c.explore_count
            policy = policy ** (1 / temperature)
            policy /= policy.sum()
            if len(actions) >= temperature_drop:
                action = root.best_child().action
            else:
                action = np.random.choice(len(policy), p=policy)
            trajectory.states.append(
                TrajectoryState(state.observation_tensor(), state.current_player(),
                                state.legal_actions_mask(), action, policy,
                                root.total_reward / root.explore_count, opponent_id=bots[2]))
            action_str = state.action_to_string(state.current_player(), action)
            actions.append(action_str)
            logger.opt_print("Player {} sampled action: {}".format(
                state.current_player(), action_str))
            state.apply_action(action)
    logger.opt_print("Next state:\n{}".format(state))

    trajectory.returns = state.returns()
    logger.print("Game {}: Returns: {}; Actions: {}".format(
        game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
    return trajectory


def update_checkpoint(logger, queue, model, az_evaluator):
    path = None
    while True:  # Get the last message, ignore intermediate ones.
        try:
            path = queue.get_nowait()
        except spawn.Empty:
            break

    if path:
        logger.print("Inference cache:", az_evaluator.current_agent.evaluator.cache_info())
        logger.print("Loading checkpoint", path)

        # bring the primary agent up to date
        latest_path = os.path.join(model._path, 'checkpoint--1')
        if os.path.exists(latest_path):
            az_evaluator.update_current_agent_to_main(latest_path)

        # Policy checkpoint (historical archive)
        if re.match(r'.*historical.*', path):
            logger.print("Adding to historical archive")
            az_evaluator.add_checkpoint_bot(path)
            # Historical checkpoint (self-play archive)
            # If we loaded a historical checkpoint, then there should also be an updated response matrix checkpoint
            # load the response matrix checkpoint
            response_matrix_path = path.replace('historical', 'response-matrix') + '.npy'
            logger.print('Loading response matrix checkpoint', response_matrix_path)
            az_evaluator.update_response_matrix(response_matrix_path)
            # check to see if we need to update the current agent (AGAIN!)
            n_cols = az_evaluator.A.shape[1]
            n_chkpts = len(az_evaluator.checkpoint_mcts_bots)
            logger.print('Number of columns in response matrix', n_cols)
            logger.print('Number of checkpoint bots', n_chkpts)
            if n_cols > n_chkpts:
                # get a list of the historical checkpoints in the save directory
                checkpoint_paths = [f.split('.')[0] for f in 
                                    os.listdir(model._path) if
                                    re.match(r'.*historical.*', f)]
                checkpoint_paths = list(set(checkpoint_paths))
                for f in checkpoint_paths:
                    full_path = os.path.join(model._path, f)
                    if full_path not in az_evaluator.checkpoint_mcts_bots:
                        az_evaluator.add_checkpoint_bot(full_path)
            logger.print("corrected number of checkpoint bots", len(az_evaluator.checkpoint_mcts_bots))

        # Policy checkpoint (novelty archive)
        elif re.match(r'.*novelty.*', path):
            logger.print('Adding to novelty archive')
            az_evaluator.add_novelty_bot(path)
            # If we loaded a novelty checkpoint, then there should also be an update response matrix checkpoint
            # load the response matrix checkpoint
            response_matrix_path = path.replace('novelty', 'response-matrix') + '.npy'
            logger.print('Loading response matrix checkpoint', response_matrix_path)
            az_evaluator.update_response_matrix(response_matrix_path)
            # check to see if we need to update the current agent (AGAIN!)
            n_rows = az_evaluator.A.shape[0]
            n_novs = len(az_evaluator.novelty_mcts_bots)
            logger.print('Number of rows in response matrix', n_rows)
            logger.print('Number of novelty bots', n_novs)
    elif path is not None:  # Empty string means stop this process.
        return False
    return True


# def update_checkpoint(logger, queue, model, az_evaluator):
#   """Read the queue for a checkpoint to load, or an exit signal."""
#   path = None
#   while True:  # Get the last message, ignore intermediate ones.
#     try:
#       path = queue.get_nowait()
#     except spawn.Empty:
#       break
#   if path:
#     logger.print("Inference cache:", az_evaluator.cache_info())
#     logger.print("Loading checkpoint", path)
#     model.load_checkpoint(path)
#     az_evaluator.clear_cache()
#   elif path is not None:  # Empty string means stop this process.
#     return False
#   return True


@watcher
def actor(*, config, game, logger, queue):
    """An actor process runner that generates games and returns trajectories."""
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Initializing bots")
    # az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    pop_az_evaluator = AZPopulationWithEvaluators(game=game, model=model, init_bot_fn=_init_bot, config=config,
                                                  k=5)
    bots = [
        _init_bot(config, game, pop_az_evaluator, False),
        _init_bot(config, game, pop_az_evaluator, False),
    ]
    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, pop_az_evaluator):
            return
        
        op_bot = bots[1]
        op_name = 'checkpoint--1'
        # sample opponent from opponents
        # if len(pop_az_evaluator.checkpoint_mcts_bots) >= 2:
        #     op = np.random.choice(list(pop_az_evaluator.checkpoint_mcts_bots.keys()))
        #     op_bot = pop_az_evaluator.checkpoint_mcts_bots[op]
        #     op_name = op

        game_bots = [bots[0], op_bot, op_name]
        queue.put(_play_game(logger, game_num, game, game_bots, config.temperature,
                             config.temperature_drop))


@watcher
def evaluator(*, game, config, logger, queue):
    """A process that plays the latest checkpoint vs standard MCTS."""
    results = Buffer(config.evaluation_window)
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Initializing bots")

    pop_az_evaluator = AZPopulationWithEvaluators(game=game, model=model, init_bot_fn=_init_bot, config=config,
                                                  k=5)
    random_evaluator = mcts.RandomRolloutEvaluator()

    for game_num in itertools.count():
        if not update_checkpoint(logger, queue, model, pop_az_evaluator):
            return

        az_player = game_num % 2
        difficulty = (game_num // 2) % config.eval_levels
        max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
        bots = [
            _init_bot(config, game, pop_az_evaluator, True),
            mcts.MCTSBot(
                game,
                config.uct_c,
                max_simulations,
                random_evaluator,
                solve=True,
                verbose=False,
                dont_return_chance_node=True)
        ]
        if az_player == 1:
            bots = list(reversed(bots))
        bots.append(f'mcts_eval_{max_simulations}')

        trajectory = _play_game(logger, game_num, game, bots, temperature=1,
                                temperature_drop=0)
        results.append(trajectory.returns[az_player])
        queue.put((difficulty, trajectory.returns[az_player]))

        logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
            trajectory.returns[az_player],
            trajectory.returns[1 - az_player],
            len(results), np.mean(results.data)))


@watcher
def learner(*, game, config, actors, evaluators, broadcast_fn, logger):
    """A learner that consumes the replay buffer and trains the network."""
    logger.also_to_stdout = True
    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                             config.nn_depth))
    logger.print("Model size:", model.num_trainable_variables, "variables")
    save_path = model.save_checkpoint(0, 'historical-checkpoint')
    logger.print("Initial checkpoint:", save_path)

    data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

    stage_count = 7
    value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
    value_predictions = [stats.BasicStats() for _ in range(stage_count)]
    game_lengths = stats.BasicStats()
    game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
    outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
    evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
    total_trajectories = 0
    extra_games = 0

    outcome_matrix = np.array([[0]])
    n_neighbors = 5
    threshold = 0.15
    novelty_distance = 0
    outcom_matrix_cardinality = 0

    # build an azpopulation object so that we can hold all the historical and novelty policies in one place
    az_evaluator = AZPopulationWithEvaluators(game=game, model=model, init_bot_fn=_init_bot, config=config,
                                              k=n_neighbors)
    population_bot = _init_bot(config, game, az_evaluator, False)

    # save the initial model as both a historical model and as a novelty model
    response_matrix_path = save_path.replace('historical', 'response-matrix') + '.npy'
    np.save(response_matrix_path, outcome_matrix)
    broadcast_fn(save_path)
    time.sleep(3)
    nov_path = model.save_checkpoint(0, 'novelty-checkpoint')
    time.sleep(3)
    broadcast_fn(nov_path)
    population_bot.evaluator.add_novelty_bot(nov_path)
    population_bot.evaluator.A = outcome_matrix

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

    def collect_but_keep_trajectories_whole():
        trajectories = []
        num_trajectories = 0
        num_states = 0
        for trajectory in trajectory_generator():
            num_states += len(trajectory.states)
            trajectories.append(trajectory)
            num_trajectories += 1
            if num_states >= learn_rate:
                break
        return trajectories

    def collect_trajectories(trajectories):
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in trajectories:
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            replay_buffer.extend(
                model_lib.TrainInput(
                    s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in trajectory.states)

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

        return num_trajectories, num_states

    def learn(step):
        """Sample from the replay buffer, update weights and save a checkpoint."""
        losses = []
        for _ in range(len(replay_buffer) // config.train_batch_size):
            data = replay_buffer.sample(config.train_batch_size)
            losses.append(model.update(data))

        # Always save a checkpoint, either for keeping or for loading the weights to
        # the actors. It only allows numbers, so use -1 as "latest".
        save_id = step if step % config.checkpoint_freq == 0 else -1
        save_moniker = 'checkpoint'
        if save_id != -1:
            save_moniker = 'historical-' + save_moniker
        save_path = model.save_checkpoint(step=save_id, model_type=save_moniker)
        losses = sum(losses, model_lib.Losses(0, 0, 0)) / len(losses)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)
        return save_path, losses

    def is_novel(A, a):
        # A is a matrix of actions
        # a is a vector of actions
        # return True if a is novel
        # calculate the knn distance between a and a's knn in A
        # if that distance is greater than some threshold, return True, dist
        # else return False, dist
        if A.size <= 1:
            return True, 1
        assert A.shape[1] == a.shape[0]
        a = np.array(a)
        dists = np.linalg.norm(A - a, axis=1)
        knn = np.argsort(dists)[:n_neighbors]
        knn_dists = dists[knn]
        avg_dist = np.mean(knn_dists)
        normalized_dist = avg_dist / (2 * np.sqrt(a.shape[0]))
        return normalized_dist >= threshold, normalized_dist

    last_time = time.time() - 60
    for step in itertools.count(1):
        for value_accuracy in value_accuracies:
            value_accuracy.reset()
        for value_prediction in value_predictions:
            value_prediction.reset()
        game_lengths.reset()
        game_lengths_hist.reset()
        outcomes.reset()

        trajectories = collect_but_keep_trajectories_whole()

        if config.novelty:

            # check if the number of historical bots is equal to the number of columns in A
            if outcome_matrix.shape[1] != len(population_bot.evaluator.checkpoint_mcts_bots):
            # load all missing checkpoint bots
                all_checkpoints = [f.split('.')[0] for f in 
                                        os.listdir(model._path) if
                                        re.match(r'.*historical.*', f)]
                checkpoint_paths = list(set(all_checkpoints))
                for f in checkpoint_paths:
                    full_path = os.path.join(model._path, f)
                    if full_path not in population_bot.evaluator.checkpoint_mcts_bots:
                        population_bot.evaluator.add_checkpoint_bot(full_path)


            # calculate novelty against the trajectories
            a_dict = collections.defaultdict(list)
            # play each of the agents in the historical-population
            for k, v in az_evaluator.checkpoint_mcts_bots.items():
                if k not in a_dict:
                    extra_games += 1
                    traj = _play_game(logger, extra_games, game, [population_bot, v, k], temperature=1, temperature_drop=0)
                    a_dict[k].append(traj.returns[1])

            a_vec = np.array(list(a_dict.values())).flatten()
            # logger.print('a_vec', a_vec)
            # logger.print('outcome_matrix', outcome_matrix)
            novelty_distance = 0
            if a_vec.shape[0] >= 2 and outcome_matrix.shape[1] >= 2:
                try:
                    is_vector_novel, novelty_distance = is_novel(outcome_matrix, a_vec)
                except AssertionError:
                    import pdb;
                    pdb.set_trace()
                if is_vector_novel:
                    try:
                        outcome_matrix = np.vstack((outcome_matrix, a_vec))
                    except ValueError:
                        import pdb;
                        pdb.set_trace()
                    for traj in trajectories:
                        traj.novelty = novelty_distance
                    # save the model checkpoint
                    save_path = model.save_checkpoint(step=step, model_type='novelty-checkpoint')
                    # save the new outcome matrix
                    response_matrix_path = save_path.replace('novelty', 'response-matrix') + '.npy'
                    np.save(response_matrix_path, outcome_matrix)
                    # wait just a moment to make sure that we finish saving the checkpoints before broadcasting
                    time.sleep(0.05)
                    broadcast_fn(save_path)
                    time.sleep(2)
                    population_bot.evaluator.add_novelty_bot(save_path)
                    population_bot.evaluator.update_response_matrix(response_matrix_path)

        num_trajectories, num_states = collect_trajectories(trajectories)
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now

        logger.print("Step:", step)
        logger.print(
            ("Collected {:5} states from {:3} games, {:.1f} states/s. "
             "{:.1f} states/(s*actor), game length: {:.1f}").format(
                num_states, num_trajectories, num_states / seconds,
                                              num_states / (config.actors * seconds),
                                              num_states / num_trajectories))
        logger.print("Buffer size: {}. States seen: {}".format(
            len(replay_buffer), replay_buffer.total_seen))

        save_path, losses = learn(step)
        if step in [1, 10]:
            save_path = model.save_checkpoint(step=step, model_type='historical-checkpoint')

        
        if config.novelty:

            if 'historical' in save_path:
                vs_checkpoint_policies = np.zeros(len(population_bot.evaluator.novelty_mcts_bots))
                if vs_checkpoint_policies.size <= 0:
                    vs_checkpoint_policies = np.zeros(1)
                # play a game against each of the novel policies
                population_bot.evaluator.add_checkpoint_bot(save_path)
                for i, (k, op) in enumerate(population_bot.evaluator.novelty_mcts_bots.items()):
                    game_bots = [population_bot.evaluator.checkpoint_mcts_bots[save_path], op, k]
                    extra_games += 1
                    trajectory = _play_game(logger, extra_games, game, game_bots, temperature=1, temperature_drop=0)
                    vs_checkpoint_policies[i] = trajectory.returns[0]
                try:
                    if outcome_matrix.shape[1] == 0:
                        outcome_matrix = np.append(outcome_matrix, vs_checkpoint_policies.reshape(-1, 1), axis=1)
                    else:
                        outcome_matrix = np.hstack((outcome_matrix, vs_checkpoint_policies.reshape(-1, 1)))
                except ValueError:
                    import pdb; pdb.set_trace()
                # save the response matrix checkpoint
                response_matrix_path = save_path.replace('historical', 'response-matrix') + '.npy'
                np.save(response_matrix_path, outcome_matrix)
                # wait just a moment to make sure that we finish saving the checkpoints before broadcasting
                time.sleep(0.1)
                population_bot.evaluator.update_response_matrix(response_matrix_path)
                # This doesn't happen super frequently, so just be willing to wait a long time here to make sure that everyong picks up this checkpoint
                broadcast_fn(save_path)
                time.sleep(5)

        save_path = model.save_checkpoint(step=-1, model_type='checkpoint')
        
        for eval_process in evaluators:
            while True:
                try:
                    difficulty, outcome = eval_process.queue.get_nowait()
                    evals[difficulty].append(outcome)
                except spawn.Empty:
                    break

        
        if config.novelty:
            if outcome_matrix.shape[1] < 2:
                outcom_matrix_cardinality = 0.5
            else:
                norms_new = np.linalg.norm(outcome_matrix, axis=1, keepdims=True)
                M_normalized_new = outcome_matrix / norms_new
                L_new = np.dot(M_normalized_new, M_normalized_new.T)
                outcom_matrix_cardinality = np.trace(np.eye(L_new.shape[0]) - np.linalg.inv(L_new + np.eye(L_new.shape[0])))
        
        batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
        batch_size_stats.add(1)
        data_log.write({
            "step": step,
            "total_states": replay_buffer.total_seen,
            "states_per_s": num_states / seconds,
            "states_per_s_actor": num_states / (config.actors * seconds),
            "total_trajectories": total_trajectories,
            "trajectories_per_s": num_trajectories / seconds,
            "queue_size": 0,  # Only available in C++.
            "game_length": game_lengths.as_dict,
            "game_length_hist": game_lengths_hist.data,
            "outcomes": outcomes.data,
            "value_accuracy": [v.as_dict for v in value_accuracies],
            "value_prediction": [v.as_dict for v in value_predictions],
            "eval": {
                "count": evals[0].total_seen,
                "results": [sum(e.data) / len(e) if e else 0 for e in evals],
            },
            "batch_size": batch_size_stats.as_dict,
            "batch_size_hist": [0, 1],
            'novelty': novelty_distance,
            'outcome_matrix_cardinality': outcom_matrix_cardinality,
            "loss": {
                "policy": losses.policy,
                "value": losses.value,
                "l2reg": losses.l2,
                "sum": losses.total,
            },
            "cache": {  # Null stats because it's hard to report between processes.
                "size": 0,
                "max_size": 0,
                "usage": 0,
                "requests": 0,
                "requests_per_s": 0,
                "hits": 0,
                "misses": 0,
                "misses_per_s": 0,
                "hit_rate": 0,
            },
        })
        logger.print()

        if config.max_steps > 0 and step >= config.max_steps:
            break

        time.sleep(1) # wait before broadcasting the checkpoint to make sure we saved any new novelty/historical policies
        broadcast_fn(save_path)


def alpha_zero(config: Config):
    """Start all the worker processes for a full alphazero setup."""
    game = pyspiel.load_game(config.game)
    if config.game == 'python_dominated_connect_four' or config.game == "python_randomized_connect_four":
        eval_game = pyspiel.load_game('connect_four')
    
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())

    print("Starting game", config.game)
    if game.num_players() != 2:
        sys.exit("AlphaZero can only handle 2-player games.")
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
        raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must have sequential turns.")
    if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("Game must be deterministic.")

    path = config.path
    if not path:
        path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
        config = config._replace(path=path)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        sys.exit("{} isn't a directory".format(path))
    print("Writing logs and checkpoints to:", path)
    print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                      config.nn_depth))

    with open(os.path.join(config.path, "config.json"), "w") as fp:
        fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

    actors = [spawn.Process(actor, kwargs={"game": game, "config": config,
                                           "num": i})
              for i in range(config.actors)]
    if config.game == 'python_dominated_connect_four' or config.game == "python_randomized_connect_four":
        evaluators = [spawn.Process(evaluator, kwargs={"game": eval_game, "config": config,
                                                     "num": i})
                         for i in range(config.evaluators)]
    else:
        evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config,
                                                   "num": i})
                      for i in range(config.evaluators)]

    def broadcast(msg):
        for proc in actors + evaluators:
            proc.queue.put(msg)

    try:
        learner(game=game, config=config, actors=actors,  # pylint: disable=missing-kwoa
                evaluators=evaluators, broadcast_fn=broadcast)
    except (KeyboardInterrupt, EOFError):
        print("Caught a KeyboardInterrupt, stopping early.")
    finally:
        broadcast("")
        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        for proc in actors:
            while proc.exitcode is None:
                while not proc.queue.empty():
                    proc.queue.get_nowait()
                proc.join(JOIN_WAIT_DELAY)
        for proc in evaluators:
            proc.join()
