import math
import random
from numpy import log as ln
from mcts.node import Node


class TreePolicy:
    def select(self, node: Node, state):
        raise NotImplementedError


class RandomTreePolicy(TreePolicy):
    def select(self, node: Node, state):
        return random.choice(node.children)

    def __str__(self):
        return "RandTree"


class UCTPolicy(TreePolicy):
    def __init__(self, exploitation_term, exploration_term):
        self.exploitation_term = exploitation_term
        self.exploration_term = exploration_term

    def select(self, node: Node):
        best_uct, child = -1e7, None
        for c in node.children:
            if c.visits > 0 and node.visits > 0:
                uct = self.exploitation_term.val(node) + self.exploration_term.val(node)
                if uct > best_uct:
                    best_uct = uct
                    child = c
        if not child:
            child = random.choice(node.children)

        return child

    def __str__(self):
        return str(self.exploitation_term) + str(self.exploration_term)