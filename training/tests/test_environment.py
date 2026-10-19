import json
from pathlib import Path

import numpy as np
import pytest

from training.gops_rl.environment import GOPSEnvironment, shuffled_prizes


FIXTURE = json.loads((Path(__file__).parents[2] / "contracts/fixtures/seed-42-match.json").read_text())


def test_seeded_shuffle_matches_typescript_contract():
    assert shuffled_prizes(FIXTURE["seed"]) == FIXTURE["prizeOrder"]


def test_complete_fixture_and_reward_symmetry():
    environment = GOPSEnvironment()
    environment.reset(FIXTURE["seed"])
    for human_bid, bot_bid in zip(FIXTURE["humanBids"], FIXTURE["botBids"], strict=True):
        environment.step(human_bid, bot_bid)
    assert environment.state.scores == (47, 44)
    assert environment.reward(0) == pytest.approx(3 / 91)
    assert environment.reward(1) == pytest.approx(-3 / 91)


def test_final_tie_leaves_full_pot_unawarded():
    environment = GOPSEnvironment()
    environment.reset(9)
    for bid in range(1, 14):
        environment.step(bid, bid)
    assert environment.state.scores == (0, 0)
    assert environment.state.carried_pot == 91
    assert environment.reward(0) == 0


def test_observation_and_legal_masks():
    environment = GOPSEnvironment()
    environment.reset(42)
    observation = environment.observe(0)
    assert observation.shape == (56,)
    assert observation.dtype == np.float32
    assert environment.legal_mask(0).sum() == 13
    environment.step(1, 13)
    assert environment.legal_mask(0)[0] == 0
    assert environment.legal_mask(1)[12] == 0


def test_invalid_bid_is_rejected_without_advancing():
    environment = GOPSEnvironment(5)
    environment.reset(1)
    with pytest.raises(ValueError):
        environment.step(6, 1)
    assert environment.state.round_number == 1

