from .uct_term import UCTTerm
import math


class PUCTTerm(UCTTerm):
    def __init__(self, exploration_constant, dirichlet_alpha=0.03, dirichlet_epsilon=0.25):
        self.exploration_constant = exploration_constant
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def val(self, node, model=None, neural_net=None, dirichlet_noise=None):
        # (1 - eps) p + eps eta
        prior = node.prior_prob
        if dirichlet_noise:
            prior = (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * dirichlet_noise

        return self.exploration_constant * prior * (math.sqrt(node.parent.visits) / (1 + node.visits))

    def init_prior(self, children, state=None, env=None, neural_net=None):
        if not children[0].prior_prob:
            _, action_probs = neural_net.evaluate_actions(env.observation(state), [c.action for c in children])
            for c, p in zip(children, action_probs):
                c.prior_prob = p

    def __str__(self):
        return "PUCT"
