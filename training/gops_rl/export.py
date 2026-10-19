from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .model import ActorCritic, OBSERVATION_SIZE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_checkpoint(checkpoint: Path, output: Path) -> tuple[float, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden_size = int(payload["config"]["hidden_size"])
    model = ActorCritic(hidden_size)
    model.load_state_dict(payload["model_state"])
    model.eval()
    sample = torch.linspace(0, 1, OBSERVATION_SIZE, dtype=torch.float32).unsqueeze(0)
    torch.onnx.export(
        model,
        (sample,),
        output,
        input_names=["observation"],
        output_names=["logits", "value"],
        dynamic_axes={"observation": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    with torch.no_grad():
        torch_logits, torch_value = model(sample)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_value = session.run(None, {"observation": sample.numpy()})
    maximum_difference = max(
        float(np.max(np.abs(torch_logits.numpy() - onnx_logits))),
        float(np.max(np.abs(torch_value.numpy() - onnx_value))),
    )
    if maximum_difference > 1e-5:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX parity failed for {checkpoint.name}: {maximum_difference}")
    return maximum_difference, int(payload["update"])


def export_release(run_dir: Path, output_dir: Path, allow_uncalibrated: bool = False) -> Path:
    evaluation_path = run_dir / "evaluation.json"
    if not evaluation_path.exists():
        raise RuntimeError("Run evaluation before exporting release models")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not evaluation["calibrated"] and not allow_uncalibrated:
        raise RuntimeError("The selected tiers do not satisfy the required 75-Elo separation")

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for difficulty in ("beginner", "casual", "skilled", "expert"):
        report = evaluation["tiers"][difficulty]
        checkpoint = run_dir / "checkpoints" / report["checkpoint"]
        model_name = f"{difficulty}-v1.onnx"
        model_path = output_dir / model_name
        parity_difference, update = _export_checkpoint(checkpoint, model_path)
        entries.append({
            "difficulty": difficulty,
            "label": difficulty.title(),
            "file": model_name,
            "sha256": _sha256(model_path),
            "modelVersion": f"1.0.0-u{update}",
            "trainingRun": run_dir.name,
            "observationVersion": 1,
            "metrics": {
                "arenaElo": report["arenaElo"],
                "baselineWinRate": report["baselineWinRate"],
                "games": report["games"],
                "onnxMaxDifference": parity_difference,
            },
        })
    manifest = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "observationSize": 56,
        "actionSize": 13,
        "calibrated": evaluation["calibrated"],
        "models": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(evaluation_path, output_dir / "evaluation.json")
    return manifest_path

