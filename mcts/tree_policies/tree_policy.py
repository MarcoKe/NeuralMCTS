import random
import numpy as np
from mcts.node import Node
from mcts.tree_policies.exploitation_terms.exploitation_term import ExploitationTerm
from mcts.tree_policies.exploration_terms.exploration_term import ExplorationTerm


class TreePolicy:
    def select(self, node: Node, state):
        raise NotImplementedError


class RandomTreePolicy(TreePolicy):
    def select(self, node: Node, state):
        return random.choice(node.children)

    def __str__(self):
        return "RandTree"


class UCTPolicy(TreePolicy):
    def __init__(self, exploitation_term: ExploitationTerm, exploration_term: ExplorationTerm, dirichlet_alpha=0.03):
        self.exploitation_term = exploitation_term
        self.exploration_term = exploration_term

    def select(self, node: Node, add_dirichlet: bool = False):
        best_uct, child = -1e7, None

        dirichlet_noise = None
        if add_dirichlet:
            dirichlet_noise = np.random.dirichlet([self.exploration_term.dirichlet_alpha]*len(node.children))

        for i, c in enumerate(node.children):
            if c.visits > 0 and node.visits > 0:
                noise = dirichlet_noise[i] if add_dirichlet else None
                uct = self.exploitation_term.val(c) + self.exploration_term.val(c, dirichlet_noise=noise)
                if uct > best_uct:
                    best_uct = uct
                    child = c
        if not child:
            child = random.choice(node.children)

        return child

    def __str__(self):
        return str(self.exploitation_term) + str(self.exploration_term)