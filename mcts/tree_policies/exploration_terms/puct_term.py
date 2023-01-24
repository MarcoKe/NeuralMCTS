from .uct_term import UCTTerm
import math

class PUCTTerm(UCTTerm):
    def __init__(self, exploration_constant, dirichlet_alpha=0.03, dirichlet_epsilon=0.25):
        self.exploration_constant = exploration_constant
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def val(self, node, add_dirichlet=False):
        prior = node.prior_prob
        # if add_dirichlet:
        #     prior

        return self.exploration_constant * prior * (math.sqrt(node.parent.visits) / (1 + node.visits))

    def __str__(self):
        return "PUCT"
