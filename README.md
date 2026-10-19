# GOPS — Game of Pure Strategy

A responsive, privacy-first web game for playing standard 13-card GOPS against four reinforcement-learning policy checkpoints. The policies are trained with masked PPO and league self-play, exported to ONNX, and evaluated entirely in the browser.

## Quick start

```powershell
npm install --prefix web
npm run dev
```

The development server prints the local URL. No backend or account is required.

## Project layout

- `web/` — React, TypeScript, Vite, browser inference, and UI tests.
- `training/` — Python/PyTorch environment, PPO self-play, evaluation, and ONNX export.
- `contracts/` — versioned model manifest and cross-runtime fixtures.

## Web commands

```powershell
npm run dev
npm run lint
npm test
npm run build
npm run test:e2e
```

## Training commands

Create a Python 3.11+ virtual environment, then install the locked dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r training/requirements.lock
.venv\Scripts\python -m pytest training/tests
.venv\Scripts\python -m training.gops_rl train --config training/configs/ppo.yaml --device auto
.venv\Scripts\python -m training.gops_rl evaluate --run-dir training/runs/latest --games 10000
.venv\Scripts\python -m training.gops_rl export --run-dir training/runs/latest --output web/public/models
```

Training saves reproducible configuration, seeds, optimizer state, checkpoints, and evaluation reports under `training/runs/`. The checked-in web models are release artifacts; routine web builds never retrain them.

## GitHub Pages

Push the repository to GitHub and enable **Settings → Pages → GitHub Actions**. Every push to `main` runs the complete quality workflow and deploys the static application after all checks pass.

## Rules

Each side bids Ace (1) through King (13) once. A diamond prize is revealed each round. The higher bid wins the current prize and any carried pot. Equal bids carry the pot forward. A pot tied on the final round is unawarded; awarded scores determine the match winner.

