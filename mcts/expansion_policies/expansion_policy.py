from ..node import Node


class ExpansionPolicy:
    def expand(self, node, state, model, **kwargs):
        new_children = []
        for a in model.legal_actions(state):
            child = Node(a, node)
            new_children.append(child)
            node.children.append(child)

        return new_children

    def __str__(self):
        return "FullExp"


