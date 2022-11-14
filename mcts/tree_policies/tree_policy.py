import math


from mcts.node import Node
import random
from numpy import log as ln


class TreePolicy:
    def select(self, node: Node, state):
        raise NotImplementedError


class RandomTreePolicy(TreePolicy):
    def select(self, node: Node, state):
        return random.choice(node.children)

    def __str__(self):
        return "RandTree"

class UCTPolicy(TreePolicy):
    def __init__(self, exploration_const):
        self.exp_const = exploration_const

    def Q(self, child):
        return child.returns / child.visits

    def U(self, node, child, state):
        return self.exp_const * math.sqrt(ln(node.visits) / child.visits)

    def select(self, node: Node, state):
        best_uct, child = -1e7, None
        for c in node.children:
            if c.visits > 0 and node.visits > 0:
                uct = self.Q(c) + self.U(node, c, state)
                if uct > best_uct:
                    best_uct = uct
                    child = c
        if not child:
            # print("here")
            child = random.choice(node.children)

        return child

    def __str__(self):
        return "UCT"