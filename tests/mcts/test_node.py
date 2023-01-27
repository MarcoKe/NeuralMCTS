from mcts.node import Node


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
