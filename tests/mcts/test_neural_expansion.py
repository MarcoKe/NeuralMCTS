from mcts.expansion_policies.neural_expansion import NeuralExpansionPolicy


def test_select_actions():
    threshold = 0.6
    expansion_policy = NeuralExpansionPolicy(threshold)

    actions = [1, 11, 15, 2]
    action_probs = [0.3, 0.2, 0.4, 0.1]
    selected_actions = list(expansion_policy.select_actions(action_probs, actions))

    assert selected_actions == [15, 1]

    threshold = 0.3
    expansion_policy = NeuralExpansionPolicy(threshold)
    selected_actions = list(expansion_policy.select_actions(action_probs, actions))

    assert selected_actions == [15]

    threshold = 0.4
    expansion_policy = NeuralExpansionPolicy(threshold)
    selected_actions = list(expansion_policy.select_actions(action_probs, actions))

    assert selected_actions == [15]

    threshold = 1.0
    expansion_policy = NeuralExpansionPolicy(threshold)
    selected_actions = list(expansion_policy.select_actions(action_probs, actions))

    assert selected_actions == [15, 1, 11, 2]

