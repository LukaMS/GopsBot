# Training and releasing policies

The trainer uses a shared actor-critic policy, masked discrete-action PPO, and an opponent league containing the current policy plus archived snapshots. The learning agent is randomly assigned a seat for every game. Rewards are the final normalized score difference, with no hand-written strategic shaping.

## Local workflow

From the repository root:

```powershell
.venv\Scripts\python -m training.gops_rl train `
  --config training/configs/ppo.yaml `
  --run-dir training/runs/my-run `
  --device auto

.venv\Scripts\python -m training.gops_rl evaluate `
  --run-dir training/runs/my-run `
  --games 10000 `
  --device auto

.venv\Scripts\python -m training.gops_rl export `
  --run-dir training/runs/my-run `
  --output web/public/models
```

Use `--resume <checkpoint>` to continue an interrupted run. Evaluation caches each completed checkpoint immediately, so large arenas can also resume without repeating completed work.

## Cloud portability

Build the same environment from the repository root:

```powershell
docker build -f training/Dockerfile -t gops-trainer .
docker run --rm -v ${PWD}/training/runs:/app/training/runs gops-trainer train --config training/configs/ppo.yaml --run-dir training/runs/cloud --device cpu
```

On a CUDA host, add `--gpus all` and select `--device cuda`. The command-line interface, configurations, checkpoints, and output layout are identical locally and in a container.

## Release gates

- Each tier is a real checkpoint using the same sampling rule; easier bots do not receive injected mistakes.
- Adjacent tiers need at least 75 estimated Elo and non-overlapping 95% bootstrap score-rate intervals.
- Expert must win at least 70% against the combined random, nearest-prize, and pot-aware suite over the release arena.
- Export fails when calibration does not pass unless an explicit development-only override is supplied.
- ONNX export compares logits and values against PyTorch and fails above `1e-5` maximum absolute error.

The checked-in `web/public/models/evaluation.json` records the release evidence. Training checkpoints and optimizer state stay outside Git because they are reproducible and much larger than the browser artifacts.

