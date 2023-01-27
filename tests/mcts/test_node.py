from mcts.node import Node


def test_is_leaf():
    n = Node(None, None)
    assert n.is_leaf()

    n.expand([1, 2, 3])
    assert not n.is_leaf()
