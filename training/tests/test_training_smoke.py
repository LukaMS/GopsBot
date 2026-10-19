from pathlib import Path

import torch

from training.gops_rl.ppo import train


def test_training_writes_resumable_checkpoints(tmp_path: Path):
    config = Path(__file__).parents[1] / "configs/smoke.yaml"
    run_dir = train(config, tmp_path / "run", "cpu")
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    assert checkpoints
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    assert payload["update"] == 2
    assert "optimizer_state" in payload

