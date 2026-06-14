from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from wizard.torchNEP.config import TrainConfig
from wizard.torchNEP.datasets.data import build_dataset
from wizard.torchNEP.datasets.dataset import collate_fn
from wizard.torchNEP.nep.model import NEP
from wizard.torchNEP.runtime.distributed import resolve_device


@dataclass
class ScalarStats:
    abs_sum: float = 0.0
    square_sum: float = 0.0
    max_abs: float = 0.0
    count: int = 0

    def add(self, diff: torch.Tensor) -> None:
        values = diff.detach().reshape(-1).float().cpu()
        if values.numel() == 0:
            return
        abs_values = torch.abs(values)
        self.abs_sum += float(torch.sum(abs_values).item())
        self.square_sum += float(torch.sum(values * values).item())
        self.max_abs = max(self.max_abs, float(torch.max(abs_values).item()))
        self.count += int(values.numel())

    def as_dict(self, prefix: str) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                f"{prefix}_mae": None,
                f"{prefix}_rmse": None,
                f"{prefix}_max_abs": None,
                f"{prefix}_count": 0,
            }
        return {
            f"{prefix}_mae": self.abs_sum / self.count,
            f"{prefix}_rmse": (self.square_sum / self.count) ** 0.5,
            f"{prefix}_max_abs": self.max_abs,
            f"{prefix}_count": self.count,
        }


@dataclass
class EvaluationSummary:
    artifact: Path
    split: str
    frames: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompareSummary:
    left: Path
    right: Path
    split: str
    frames: int
    metrics: dict[str, Any] = field(default_factory=dict)


def evaluate_artifact(
    config: TrainConfig,
    artifact: str | Path = "checkpoints/best.pt",
    split: str | None = None,
    device: str | torch.device | None = None,
) -> EvaluationSummary:
    split = resolve_split(config, split)
    artifact_path = require_existing(resolve_artifact_path(config, artifact))
    resolved_device = resolve_device(device or config.runtime.device)
    model = load_artifact_model(artifact_path, resolved_device)
    loader = build_eval_loader(config, split)

    stats = {
        "energy": ScalarStats(),
        "forces": ScalarStats(),
        "virial": ScalarStats(),
    }
    model.eval()
    for batch in loader:
        batch = prepare_batch(batch, resolved_device)
        batch["compute_virial"] = "virial" in batch
        prediction = model(batch)
        accumulate_label_metrics(stats, prediction, batch)

    metrics: dict[str, Any] = {}
    for name, stat in stats.items():
        metrics.update(stat.as_dict(name))
    return EvaluationSummary(artifact=artifact_path, split=split, frames=len(loader.dataset), metrics=metrics)


def compare_artifacts(
    config: TrainConfig,
    left: str | Path = "exports/nep.txt",
    right: str | Path = "checkpoints/best.pt",
    split: str | None = None,
    device: str | torch.device | None = None,
) -> CompareSummary:
    split = resolve_split(config, split)
    left_path = require_existing(resolve_artifact_path(config, left))
    right_path = require_existing(resolve_artifact_path(config, right))
    resolved_device = resolve_device(device or config.runtime.device)
    left_model = load_artifact_model(left_path, resolved_device)
    right_model = load_artifact_model(right_path, resolved_device)
    loader = build_eval_loader(config, split)

    stats = {
        "energy": ScalarStats(),
        "forces": ScalarStats(),
        "virial": ScalarStats(),
    }
    left_model.eval()
    right_model.eval()
    for batch in loader:
        batch = prepare_batch(batch, resolved_device)
        batch["compute_virial"] = True
        left_prediction = left_model(batch)
        right_prediction = right_model(batch)
        accumulate_prediction_diffs(stats, left_prediction, right_prediction, batch)

    metrics: dict[str, Any] = {}
    for name, stat in stats.items():
        metrics.update(stat.as_dict(name))
    return CompareSummary(left=left_path, right=right_path, split=split, frames=len(loader.dataset), metrics=metrics)


def export_artifact(
    config: TrainConfig,
    artifact: str | Path = "checkpoints/best.pt",
    output: str | Path = "exports/nep.txt",
    device: str | torch.device | None = None,
) -> Path:
    artifact_path = require_existing(resolve_artifact_path(config, artifact))
    output_path = resolve_output_path(config, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device or config.runtime.device)
    model = load_artifact_model(artifact_path, resolved_device)
    model.save_to_nep_format(output_path)
    return output_path


def load_artifact_model(path: str | Path, device: torch.device | None = None) -> NEP:
    path = Path(path)
    if path.suffix == ".txt":
        model = NEP.from_nep_txt(path, device=device)
    else:
        model = NEP.from_checkpoint(path, device=device)
    model.eval()
    return model


def build_eval_loader(config: TrainConfig, split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )


def accumulate_label_metrics(stats: dict[str, ScalarStats], prediction: dict, batch: dict) -> None:
    if "energy" in prediction and "energy" in batch and "is_energy" in batch:
        mask = batch["is_energy"]
        if torch.any(mask):
            n_atoms = batch["n_atoms_per_structure"][mask].float()
            diff = prediction["energy"][mask] / n_atoms - batch["energy"][mask] / n_atoms
            stats["energy"].add(diff)

    if "forces" in prediction and "forces" in batch:
        stats["forces"].add(prediction["forces"] - batch["forces"])

    if "virial" in prediction and "virial" in batch and "is_virial" in batch:
        mask = batch["is_virial"]
        if torch.any(mask):
            n_atoms = batch["n_atoms_per_structure"][mask].float().unsqueeze(-1)
            diff = prediction["virial"][mask] / n_atoms - batch["virial"][mask] / n_atoms
            stats["virial"].add(diff)


def accumulate_prediction_diffs(stats: dict[str, ScalarStats], left: dict, right: dict, batch: dict) -> None:
    if "energy" in left and "energy" in right:
        n_atoms = batch["n_atoms_per_structure"].float()
        stats["energy"].add((left["energy"] - right["energy"]) / n_atoms)
    if "forces" in left and "forces" in right:
        stats["forces"].add(left["forces"] - right["forces"])
    if "virial" in left and "virial" in right:
        n_atoms = batch["n_atoms_per_structure"].float().unsqueeze(-1)
        stats["virial"].add((left["virial"] - right["virial"]) / n_atoms)


def prepare_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def resolve_split(config: TrainConfig, split: str | None) -> str:
    if split is not None:
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test.")
        return split
    if config.test_path is not None and config.test_path.exists():
        return "test"
    return "train"


def resolve_artifact_path(config: TrainConfig, artifact: str | Path) -> Path:
    path = Path(artifact)
    if path.is_absolute():
        return path

    candidates: list[Path] = []
    if path.parts and path.parts[0] in {"checkpoints", "exports"}:
        candidates.append(config.run_dir / path)
    elif path.suffix == ".pt":
        candidates.append(config.checkpoints_dir / path)
    elif path.suffix == ".txt":
        candidates.append(config.exports_dir / path)
        candidates.append(config.run_dir / path)
    candidates.append(config.run_dir / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(config: TrainConfig, output: str | Path) -> Path:
    path = Path(output)
    return path if path.is_absolute() else config.run_dir / path


def require_existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def format_summary(summary: EvaluationSummary | CompareSummary) -> str:
    if isinstance(summary, CompareSummary):
        lines = [
            f"Compare split={summary.split} frames={summary.frames}",
            f"  left:  {summary.left}",
            f"  right: {summary.right}",
        ]
    else:
        lines = [
            f"Evaluate split={summary.split} frames={summary.frames}",
            f"  artifact: {summary.artifact}",
        ]
    for group in ("energy", "forces", "virial"):
        lines.append(_format_metric_group(group, summary.metrics))
    return "\n".join(lines)


def _format_metric_group(group: str, metrics: dict[str, Any]) -> str:
    mae = metrics.get(f"{group}_mae")
    rmse = metrics.get(f"{group}_rmse")
    max_abs = metrics.get(f"{group}_max_abs")
    count = metrics.get(f"{group}_count", 0)
    if count == 0 or mae is None or rmse is None or max_abs is None:
        return f"  {group}: n/a"
    unit = {
        "energy": "eV/atom",
        "forces": "eV/A",
        "virial": "eV/atom",
    }[group]
    return (
        f"  {group}: mae={mae:.6e} {unit}, "
        f"rmse={rmse:.6e} {unit}, max={max_abs:.6e} {unit}, count={count}"
    )
