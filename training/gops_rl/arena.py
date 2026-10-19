from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .environment import GOPSEnvironment, GameState
from .model import ActorCritic


class Agent(Protocol):
    name: str

    def choose_batch(
        self,
        observations: np.ndarray,
        legal_masks: np.ndarray,
        states: list[GameState],
        players: np.ndarray,
    ) -> np.ndarray: ...


class PolicyAgent:
    def __init__(self, checkpoint: Path, device: torch.device, seed: int = 0):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        config = payload["config"]
        self.model = ActorCritic(int(config["hidden_size"])).to(device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.device = device
        self.name = checkpoint.stem
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.update = int(payload["update"])

    def choose_batch(self, observations, legal_masks, states, players) -> np.ndarray:
        del states, players
        with torch.no_grad():
            observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(legal_masks, dtype=torch.float32, device=self.device)
            distribution, _ = self.model.distribution(observation, mask)
            actions = torch.multinomial(distribution.probs, 1, generator=self.generator).squeeze(1)
        return actions.cpu().numpy() + 1


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.random = np.random.default_rng(seed)

    def choose_batch(self, observations, legal_masks, states, players) -> np.ndarray:
        del observations, states, players
        choices = []
        for mask in legal_masks:
            legal = np.flatnonzero(mask) + 1
            choices.append(int(self.random.choice(legal)))
        return np.asarray(choices)


class NearestPrizeAgent:
    name = "nearest-prize"

    def choose_batch(self, observations, legal_masks, states, players) -> np.ndarray:
        del observations, players
        choices = []
        for mask, state in zip(legal_masks, states, strict=True):
            legal = np.flatnonzero(mask) + 1
            target = int(state.current_prize or 1)
            choices.append(int(min(legal, key=lambda card: (abs(card - target), card))))
        return np.asarray(choices)


class PotAwareAgent:
    name = "pot-aware"

    def choose_batch(self, observations, legal_masks, states, players) -> np.ndarray:
        del observations, players
        choices = []
        for mask, state in zip(legal_masks, states, strict=True):
            legal = np.flatnonzero(mask) + 1
            pot = state.carried_pot + int(state.current_prize or 0)
            target = min(state.deck_size, max(1, round(pot * state.deck_size / max(1, state.deck_size * 2))))
            choices.append(int(min(legal, key=lambda card: (abs(card - target), -card))))
        return np.asarray(choices)


@dataclass(frozen=True)
class ArenaResult:
    wins: int
    losses: int
    draws: int
    mean_score_difference: float

    @property
    def score_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return (self.wins + 0.5 * self.draws) / total if total else 0.5


def play_matchup(agent_zero: Agent, agent_one: Agent, games: int, seed: int, deck_size: int = 13) -> ArenaResult:
    environments = [GOPSEnvironment(deck_size) for _ in range(games)]
    random = np.random.default_rng(seed)
    for environment in environments:
        environment.reset(int(random.integers(0, 2**32, dtype=np.uint64)))

    for _ in range(deck_size):
        states = [environment.state for environment in environments]
        if any(state is None for state in states):
            raise RuntimeError("Arena environment was not initialized")
        typed_states = [state for state in states if state is not None]
        observations_zero = np.stack([environment.observe(0) for environment in environments])
        observations_one = np.stack([environment.observe(1) for environment in environments])
        masks_zero = np.stack([environment.legal_mask(0) for environment in environments])
        masks_one = np.stack([environment.legal_mask(1) for environment in environments])
        players_zero = np.zeros(games, dtype=np.int64)
        players_one = np.ones(games, dtype=np.int64)
        bids_zero = agent_zero.choose_batch(observations_zero, masks_zero, typed_states, players_zero)
        bids_one = agent_one.choose_batch(observations_one, masks_one, typed_states, players_one)
        for environment, bid_zero, bid_one in zip(environments, bids_zero, bids_one, strict=True):
            environment.step(int(bid_zero), int(bid_one))

    differences = np.asarray([environment.state.scores[0] - environment.state.scores[1] for environment in environments])
    return ArenaResult(
        wins=int(np.sum(differences > 0)),
        losses=int(np.sum(differences < 0)),
        draws=int(np.sum(differences == 0)),
        mean_score_difference=float(np.mean(differences)),
    )


def _elo_from_score(score: float) -> float:
    bounded = min(0.999, max(0.001, score))
    return 400.0 * math.log10(bounded / (1.0 - bounded))


def bootstrap_interval_from_counts(wins: int, losses: int, draws: int, seed: int = 0, samples: int = 2000) -> tuple[float, float]:
    random = np.random.default_rng(seed)
    total = wins + losses + draws
    probabilities = np.asarray([wins, losses, draws], dtype=np.float64) / total
    sampled_counts = random.multinomial(total, probabilities, size=samples)
    rates = (sampled_counts[:, 0] + 0.5 * sampled_counts[:, 2]) / total
    return float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))


def evaluate_run(
    run_dir: Path,
    games: int,
    device: torch.device,
    seed: int = 9182,
    requested_updates: set[int] | None = None,
) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*.pt"))
    run_metadata_path = run_dir / "run.json"
    if requested_updates:
        checkpoints = [path for path in checkpoints if int(path.stem.split("-")[-1]) in requested_updates]
    elif run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        tier_updates = {int(update) for update in run_metadata.get("config", {}).get("tier_updates", [])}
        tier_checkpoints = [path for path in checkpoints if int(path.stem.split("-")[-1]) in tier_updates]
        if len(tier_checkpoints) >= 4:
            checkpoints = tier_checkpoints
    if len(checkpoints) < 4:
        raise RuntimeError("At least four checkpoints are required to calibrate the difficulty tiers")
    baselines: list[Agent] = [RandomAgent(seed + 1), NearestPrizeAgent(), PotAwareAgent()]
    cache_path = run_dir / f"evaluation-cache-{games}-{seed}.json"
    cached_reports = {}
    if cache_path.exists():
        cached_reports = {
            item["checkpoint"]: item
            for item in json.loads(cache_path.read_text(encoding="utf-8")).get("candidates", [])
        }
    reports = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        if checkpoint.name in cached_reports:
            reports.append(cached_reports[checkpoint.name])
            print(
                f"{checkpoint.name}: baseline={reports[-1]['baselineWinRate']:.3f} "
                f"elo={reports[-1]['arenaElo']:+.1f} (cached)",
                flush=True,
            )
            continue
        policy = PolicyAgent(checkpoint, device, seed + checkpoint_index)
        baseline_reports = []
        total_wins = total_losses = total_draws = 0
        score_rates = []
        for baseline_index, baseline in enumerate(baselines):
            first = play_matchup(policy, baseline, games, seed + checkpoint_index * 100 + baseline_index)
            second = play_matchup(baseline, policy, games, seed + 50000 + checkpoint_index * 100 + baseline_index)
            wins = first.wins + second.losses
            losses = first.losses + second.wins
            draws = first.draws + second.draws
            rate = (wins + 0.5 * draws) / (wins + losses + draws)
            total_wins += wins
            total_losses += losses
            total_draws += draws
            score_rates.append(rate)
            baseline_reports.append({
                "opponent": baseline.name,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "scoreRate": rate,
            })
        combined_rate = (total_wins + 0.5 * total_draws) / (total_wins + total_losses + total_draws)
        lower_rate, upper_rate = bootstrap_interval_from_counts(
            total_wins,
            total_losses,
            total_draws,
            seed + checkpoint_index,
        )
        reports.append({
            "checkpoint": checkpoint.name,
            "update": policy.update,
            "arenaElo": round(_elo_from_score(float(np.mean(score_rates))), 2),
            "baselineWinRate": round(combined_rate, 6),
            "games": 2 * games * len(baselines),
            "scoreRate95PercentCI": [round(lower_rate, 6), round(upper_rate, 6)],
            "matchups": baseline_reports,
            "exploitabilityProbe": round(max(0.0, 0.5 - min(score_rates)), 6),
        })
        cache_path.write_text(json.dumps({"candidates": reports}, indent=2), encoding="utf-8")
        print(f"{checkpoint.name}: baseline={combined_rate:.3f} elo={reports[-1]['arenaElo']:+.1f}", flush=True)

    ranked = sorted(reports, key=lambda report: (report["arenaElo"], report["update"]))
    selected_indices = np.linspace(0, len(ranked) - 1, 4).round().astype(int)
    selected = [ranked[index] for index in selected_indices]
    labels = ["beginner", "casual", "skilled", "expert"]
    tiers = {label: report for label, report in zip(labels, selected, strict=True)}
    elo_values = [float(tiers[label]["arenaElo"]) for label in labels]
    confidence_intervals = [tiers[label]["scoreRate95PercentCI"] for label in labels]
    calibrated = all(next_elo - elo >= 75 for elo, next_elo in zip(elo_values, elo_values[1:])) and all(
        current[1] < following[0]
        for current, following in zip(confidence_intervals, confidence_intervals[1:])
    )
    report = {
        "schemaVersion": 1,
        "seed": seed,
        "gamesPerSeatAndBaseline": games,
        "calibrated": calibrated,
        "requiredEloGap": 75,
        "candidates": reports,
        "tiers": tiers,
    }
    output = run_dir / "evaluation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
