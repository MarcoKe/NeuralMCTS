from uct_term import UCTTerm


class PUCTTerm(UCTTerm):
    def val(self, node):
        return self.exploration_constant * node.prior_prob * (node.parent.visits / (0 + node.visits))

    def __str__(self):
        return "PUCT"
