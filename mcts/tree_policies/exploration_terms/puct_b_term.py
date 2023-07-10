from .puct_term import PUCTTerm


class PUCTBTerm(PUCTTerm):
    def __init__(self, exploration_constant, best_value_constant, dirichlet_alpha=0.03, dirichlet_epsilon=0.25):
        super().__init__(exploration_constant, dirichlet_alpha, dirichlet_epsilon)
        self.best_value_constant = best_value_constant

    def val(self, node, dirichlet_noise=None):
        puct = super().val(node, dirichlet_noise)

        puct_b = puct + (self.best_value_constant * self.node.max_return)
        return puct_b

    def __str__(self):
        return "PUCT_B"
