from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

MAX_RANK = 13
OBSERVATION_SIZE = 56
Player = Literal[0, 1]


class Mulberry32:
    """The same seeded generator used by the TypeScript rules engine."""

    def __init__(self, seed: int):
        self.value = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.value = (self.value + 0x6D2B79F5) & 0xFFFFFFFF
        value = self.value
        value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFFFFFF
        value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        value &= 0xFFFFFFFF
        return ((value ^ (value >> 14)) & 0xFFFFFFFF) / 4294967296


def shuffled_prizes(seed: int, deck_size: int = MAX_RANK) -> list[int]:
    random = Mulberry32(seed)
    prizes = list(range(1, deck_size + 1))
    for index in range(len(prizes) - 1, 0, -1):
        swap_index = int(random.random() * (index + 1))
        prizes[index], prizes[swap_index] = prizes[swap_index], prizes[index]
    return prizes


@dataclass(frozen=True)
class RoundTransition:
    round_number: int
    prize: int
    pot_value: int
    bids: tuple[int, int]
    winner: int | None
    awarded: int
    carried_after: int
    scores: tuple[int, int]


@dataclass(frozen=True)
class GameState:
    seed: int
    deck_size: int
    round_number: int
    remaining_bids: tuple[tuple[int, ...], tuple[int, ...]]
    remaining_prizes: tuple[int, ...]
    revealed_prizes: tuple[int, ...]
    current_prize: int | None
    carried_pot: int
    scores: tuple[int, int]
    history: tuple[RoundTransition, ...]
    complete: bool


class GOPSEnvironment:
    def __init__(self, deck_size: int = MAX_RANK):
        if not 2 <= deck_size <= MAX_RANK:
            raise ValueError("deck_size must be between 2 and 13")
        self.deck_size = deck_size
        self.state: GameState | None = None

    @property
    def maximum_score(self) -> int:
        return self.deck_size * (self.deck_size + 1) // 2

    def reset(self, seed: int) -> GameState:
        prizes = shuffled_prizes(seed, self.deck_size)
        cards = tuple(range(1, self.deck_size + 1))
        self.state = GameState(
            seed=seed & 0xFFFFFFFF,
            deck_size=self.deck_size,
            round_number=1,
            remaining_bids=(cards, cards),
            remaining_prizes=tuple(prizes[1:]),
            revealed_prizes=(prizes[0],),
            current_prize=prizes[0],
            carried_pot=0,
            scores=(0, 0),
            history=(),
            complete=False,
        )
        return self.state

    def legal_mask(self, player: Player) -> np.ndarray:
        state = self._require_state()
        result = np.zeros(MAX_RANK, dtype=np.float32)
        result[np.asarray(state.remaining_bids[player], dtype=np.int64) - 1] = 1.0
        return result

    def observe(self, player: Player) -> np.ndarray:
        state = self._require_state()
        if state.complete or state.current_prize is None:
            raise RuntimeError("Cannot observe a completed game")
        opponent: Player = 1 if player == 0 else 0

        def mask(values: tuple[int, ...]) -> np.ndarray:
            result = np.zeros(MAX_RANK, dtype=np.float32)
            if values:
                result[np.asarray(values, dtype=np.int64) - 1] = 1.0
            return result

        current = np.zeros(MAX_RANK, dtype=np.float32)
        current[state.current_prize - 1] = 1.0
        denominator = float(self.maximum_score)
        round_denominator = float(max(1, self.deck_size - 1))
        return np.concatenate(
            (
                mask(state.remaining_bids[player]),
                mask(state.remaining_bids[opponent]),
                mask(state.remaining_prizes),
                current,
                np.asarray(
                    [
                        state.carried_pot / denominator,
                        state.scores[player] / denominator,
                        state.scores[opponent] / denominator,
                        (state.round_number - 1) / round_denominator,
                    ],
                    dtype=np.float32,
                ),
            ),
            dtype=np.float32,
        )

    def step(self, bid_player_zero: int, bid_player_one: int) -> GameState:
        state = self._require_state()
        if state.complete or state.current_prize is None:
            raise RuntimeError("This game is already complete")
        bids = (int(bid_player_zero), int(bid_player_one))
        for player, bid in enumerate(bids):
            if bid not in state.remaining_bids[player]:
                raise ValueError(f"Bid {bid} is not available for player {player}")

        remaining = tuple(
            tuple(card for card in state.remaining_bids[player] if card != bids[player])
            for player in (0, 1)
        )
        pot_value = state.carried_pot + state.current_prize
        winner = None if bids[0] == bids[1] else (0 if bids[0] > bids[1] else 1)
        scores = list(state.scores)
        if winner is not None:
            scores[winner] += pot_value
        carried_after = pot_value if winner is None else 0
        transition = RoundTransition(
            round_number=state.round_number,
            prize=state.current_prize,
            pot_value=pot_value,
            bids=bids,
            winner=winner,
            awarded=0 if winner is None else pot_value,
            carried_after=carried_after,
            scores=(scores[0], scores[1]),
        )
        complete = state.round_number == self.deck_size
        if complete:
            self.state = replace(
                state,
                remaining_bids=remaining,
                remaining_prizes=(),
                current_prize=None,
                carried_pot=carried_after,
                scores=(scores[0], scores[1]),
                history=(*state.history, transition),
                complete=True,
            )
        else:
            next_prize = state.remaining_prizes[0]
            self.state = replace(
                state,
                round_number=state.round_number + 1,
                remaining_bids=remaining,
                remaining_prizes=state.remaining_prizes[1:],
                revealed_prizes=(*state.revealed_prizes, next_prize),
                current_prize=next_prize,
                carried_pot=carried_after,
                scores=(scores[0], scores[1]),
                history=(*state.history, transition),
            )
        return self.state

    def reward(self, player: Player) -> float:
        state = self._require_state()
        if not state.complete:
            return 0.0
        opponent = 1 if player == 0 else 0
        return (state.scores[player] - state.scores[opponent]) / self.maximum_score

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Call reset before using the environment")
        return self.state

