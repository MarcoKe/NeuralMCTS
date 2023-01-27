from mcts.mcts_main import MCTSAgent
from pytest import approx


def test_exp_visit_counts():
    counts = [10, 0, 0]
    total = 10

    probs = MCTSAgent.exponentiated_visit_counts(counts, total, temperature=1.0)
    assert probs == approx([1, 0, 0])

    probs = MCTSAgent.exponentiated_visit_counts(counts, total, temperature=0.0)
    assert probs == approx([1/3, 1/3, 1/3])
