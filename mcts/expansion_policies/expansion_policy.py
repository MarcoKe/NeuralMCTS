from ..node import Node


class ExpansionPolicy:
    def __init__(self, model):
        self.model = model

    def expand(self, node, state):
        new_children = []
        for a in self.model.legal_actions(state):
            child = Node(a, node)
            new_children.append(child)
            node.children.append(child)

        return new_children


