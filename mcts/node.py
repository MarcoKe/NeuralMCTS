import math

class Node:
    def __init__(self, action, parent):
        self.action, self.parent, self.children = action, parent, [] # action is the action that led to this node from the parent
        self.visits, self.returns, self.max_return, self.prior_prob = 0, math.inf, -math.inf, None

    def expand(self, actions):
        for a in actions:
            child = Node(a, self)
            self.children.append(child)

    def update(self, r):
        if self.visits == 0:
            self.returns = 0  # we do not want the initial math.inf to influence the estimates
        self.visits += 1
        self.returns += r
        self.max_return = r if r > self.max_return else self.max_return

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        if self.parent:
            return False

        return True

    def has_parent(self):
        return self.parent is not None

    def select_best_action(self, mode='mean'):
        best_action, best_value = 0, -math.inf
        for c in self.children:
            if c.visits > 0:
                if mode == 'mean':
                    val = c.returns / c.visits
                else:
                    val = c.max_return
                if val > best_value:
                    best_action = c.action
                    best_value = val

        return best_action, best_value