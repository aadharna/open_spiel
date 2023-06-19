import math

from open_spiel.python.algorithms import mcts

def ppuct_value(node:mcts.SearchNode, parent_explore_count, uct_c):
    """Returns the PUCT value of child."""
    if node.outcome is not None:
        return node.outcome[node.player]

    return ((node.explore_count and node.total_reward / node.explore_count) +
            uct_c * node.prior * math.sqrt(parent_explore_count) /
            (node.explore_count + 1))