
class Node:
    def __init__(self, action, parent):
        self.action, self.parent, self.children = action, parent, []  # action is the action that led to this node from the parent
        self.visits, self.returns, self.max_return = 0, 0, -1e7
        self.action_probs = []

    def expand(self, actions):
        for a in actions:
            child = Node(a, self)
            self.children.append(child)

    def update(self, r):
        self.visits += 1
        self.returns += r
        self.max_return = r if r > self.max_return else self.max_return

    def is_leaf(self):
        return len(self.children) == 0

    def has_parent(self):
        return self.parent is not None

    def select_best_action(self, mode='mean'):
        best_action, best_value = 0, -1e7
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