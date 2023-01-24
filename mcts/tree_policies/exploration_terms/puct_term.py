from .uct_term import UCTTerm
import math

class PUCTTerm(UCTTerm):
    def val(self, node):
        return self.exploration_constant * node.prior_prob * (math.sqrt(node.parent.visits) / (1 + node.visits))

    def __str__(self):
        return "PUCT"
