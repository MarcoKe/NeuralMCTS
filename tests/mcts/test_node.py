from mcts.node import Node
from pytest import approx


def test_is_leaf():
    n = Node(None, None)
    assert n.is_leaf()

    n.expand([1, 2, 3])
    assert not n.is_leaf()


def test_update():
    n = Node(None, None)
    n.update(5)

    assert n.returns == 5
    assert n.visits == 1

    n.update(10)
    assert n.returns == 15
    assert n.visits == 2
    

def test_is_root():
    n = Node(None, None)
    assert n.is_root()

    n.expand([1])
    assert not n.children[0].is_root()


def backpropagate(n, value):
    while n.has_parent():
        n.update(value)
        n = n.parent
    n.update(value)


def test_construction():
    root = Node(None, None)
    l11 = Node(1, root)
    l12 = Node(2, root)
    root.children.append(l11)
    root.children.append(l12)

    l21 = Node(3, l11)
    l22 = Node(4, l11)
    l11.children.append(l21)
    l11.children.append(l22)

    l31 = Node(5, l22)
    l32 = Node(6, l22)
    l22.children.append(l31)
    l22.children.append(l32)

    assert root.visits == 0
    assert l31.visits == 0

    backpropagate(l21, 0)
    backpropagate(l31, 0)
    backpropagate(l32, 10)
    backpropagate(l12, 1)

    assert root.max_return == 10
    assert root.returns == 11
    assert root.value() == approx(11/4)
    assert l22.visits == 2

