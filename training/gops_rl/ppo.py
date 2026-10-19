from __future__ import annotations

import copy
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .environment import GOPSEnvironment
from .model import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 20261019
    deck_size: int = 13
    hidden_size: int = 128
    updates: int = 80
    episodes_per_update: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.02
    max_grad_norm: float = 0.5
    historical_opponent_probability: float = 0.20
    snapshot_interval: int = 5
    tier_updates: tuple[int, ...] = (5, 15, 35, 80)

    @classmethod
    def from_yaml(cls, path: Path) -> "PPOConfig":
        with path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        values["tier_updates"] = tuple(values.get("tier_updates", cls.tier_updates))
        return cls(**values)


@dataclass
class Rollout:
    observations: torch.Tensor
    legal_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def _load_opponent(path: Path, hidden_size: int, device: torch.device) -> ActorCritic:
    payload = torch.load(path, map_location=device, weights_only=False)
    opponent = ActorCritic(hidden_size).to(device)
    opponent.load_state_dict(payload["model_state"])
    opponent.eval()
    return opponent


def collect_rollout(
    model: ActorCritic,
    opponent: ActorCritic,
    config: PPOConfig,
    device: torch.device,
    random_generator: np.random.Generator,
) -> Rollout:
    environments = [GOPSEnvironment(config.deck_size) for _ in range(config.episodes_per_update)]
    learner_seats = random_generator.integers(0, 2, size=config.episodes_per_update)
    for environment in environments:
        environment.reset(int(random_generator.integers(0, 2**32, dtype=np.uint64)))

    observations_by_step: list[torch.Tensor] = []
    masks_by_step: list[torch.Tensor] = []
    actions_by_step: list[torch.Tensor] = []
    log_probabilities_by_step: list[torch.Tensor] = []
    values_by_step: list[torch.Tensor] = []

    model.eval()
    opponent.eval()
    for _ in range(config.deck_size):
        learner_observations = np.stack(
            [environment.observe(int(seat)) for environment, seat in zip(environments, learner_seats, strict=True)]
        )
        learner_masks = np.stack(
            [environment.legal_mask(int(seat)) for environment, seat in zip(environments, learner_seats, strict=True)]
        )
        opponent_observations = np.stack(
            [environment.observe(int(1 - seat)) for environment, seat in zip(environments, learner_seats, strict=True)]
        )
        opponent_masks = np.stack(
            [environment.legal_mask(int(1 - seat)) for environment, seat in zip(environments, learner_seats, strict=True)]
        )

        observation_tensor = torch.as_tensor(learner_observations, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(learner_masks, dtype=torch.float32, device=device)
        opponent_observation_tensor = torch.as_tensor(opponent_observations, dtype=torch.float32, device=device)
        opponent_mask_tensor = torch.as_tensor(opponent_masks, dtype=torch.float32, device=device)
        with torch.no_grad():
            distribution, values = model.distribution(observation_tensor, mask_tensor)
            actions = distribution.sample()
            log_probabilities = distribution.log_prob(actions)
            opponent_distribution, _ = opponent.distribution(opponent_observation_tensor, opponent_mask_tensor)
            opponent_actions = opponent_distribution.sample()

        for index, environment in enumerate(environments):
            learner_bid = int(actions[index].item()) + 1
            opponent_bid = int(opponent_actions[index].item()) + 1
            if learner_seats[index] == 0:
                environment.step(learner_bid, opponent_bid)
            else:
                environment.step(opponent_bid, learner_bid)

        observations_by_step.append(observation_tensor.cpu())
        masks_by_step.append(mask_tensor.cpu())
        actions_by_step.append(actions.cpu())
        log_probabilities_by_step.append(log_probabilities.cpu())
        values_by_step.append(values.cpu())

    final_rewards = torch.as_tensor(
        [environment.reward(int(seat)) for environment, seat in zip(environments, learner_seats, strict=True)],
        dtype=torch.float32,
    )
    returns = final_rewards.unsqueeze(0).expand(config.deck_size, -1).reshape(-1)
    return Rollout(
        observations=torch.stack(observations_by_step).reshape(-1, 56),
        legal_masks=torch.stack(masks_by_step).reshape(-1, 13),
        actions=torch.stack(actions_by_step).reshape(-1),
        old_log_probabilities=torch.stack(log_probabilities_by_step).reshape(-1),
        returns=returns,
        old_values=torch.stack(values_by_step).reshape(-1),
    )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    advantages = rollout.returns - rollout.old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    sample_count = rollout.actions.shape[0]
    latest: dict[str, float] = {}

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            observations = rollout.observations[indices].to(device)
            masks = rollout.legal_masks[indices].to(device)
            actions = rollout.actions[indices].to(device)
            old_log_probabilities = rollout.old_log_probabilities[indices].to(device)
            batch_returns = rollout.returns[indices].to(device)
            batch_advantages = advantages[indices].to(device)

            distribution, values = model.distribution(observations, masks)
            log_probabilities = distribution.log_prob(actions)
            ratio = (log_probabilities - old_log_probabilities).exp()
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch_returns)
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            latest = {
                "loss": float(loss.detach().cpu()),
                "policyLoss": float(policy_loss.detach().cpu()),
                "valueLoss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "meanReturn": float(rollout.returns.mean()),
            }
    return latest


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    update: int,
    config: PPOConfig,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "update": update,
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )


def train(config_path: Path, run_dir: Path, requested_device: str = "auto", resume: Path | None = None) -> Path:
    config = PPOConfig.from_yaml(config_path)
    device = resolve_device(requested_device)
    set_global_seed(config.seed)
    random_generator = np.random.default_rng(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    model = ActorCritic(config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    starting_update = 0
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        starting_update = int(payload["update"])

    metadata: dict[str, Any] = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torchVersion": torch.__version__,
        "numpyVersion": np.__version__,
        "gitRevision": _git_revision(),
        "config": asdict(config),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path = run_dir / "metrics.jsonl"
    league: list[Path] = sorted(checkpoints_dir.glob("checkpoint-*.pt"))

    for update in range(starting_update + 1, config.updates + 1):
        opponent = copy.deepcopy(model).eval()
        if league and random_generator.random() < config.historical_opponent_probability:
            opponent = _load_opponent(Path(random_generator.choice(league)), config.hidden_size, device)
        rollout = collect_rollout(model, opponent, config, device, random_generator)
        metrics = ppo_update(model, optimizer, rollout, config, device)
        metrics["update"] = update
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        should_snapshot = update % config.snapshot_interval == 0 or update in config.tier_updates or update == config.updates
        if should_snapshot:
            checkpoint = checkpoints_dir / f"checkpoint-{update:04d}.pt"
            save_checkpoint(checkpoint, model, optimizer, update, config, metrics)
            league.append(checkpoint)

        if update == 1 or update % 5 == 0 or update == config.updates:
            print(
                f"update={update:04d} return={metrics['meanReturn']:+.3f} "
                f"entropy={metrics['entropy']:.3f} loss={metrics['loss']:.3f}",
                flush=True,
            )

    return run_dir

