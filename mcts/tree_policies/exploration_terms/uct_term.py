import math
from numpy import log as ln
from .exploration_term import ExplorationTerm


class UCTTerm(ExplorationTerm):
    def __init__(self, exploration_constant):
        self.exploration_constant = exploration_constant

    def val(self, node):
        return self.exploration_constant * math.sqrt(ln(node.parent.visits) / node.visits)

    def __str__(self):
        return "UCT"