import torch

from training.gops_rl.model import ActorCritic


def test_action_mask_makes_spent_cards_impossible():
    model = ActorCritic(hidden_size=16)
    observation = torch.zeros((64, 56))
    mask = torch.zeros((64, 13))
    mask[:, 4] = 1
    distribution, value = model.distribution(observation, mask)
    assert torch.all(distribution.sample() == 4)
    assert value.shape == (64,)


def test_policy_contract_shapes():
    model = ActorCritic(hidden_size=16)
    logits, value = model(torch.zeros((3, 56)))
    assert logits.shape == (3, 13)
    assert value.shape == (3,)

