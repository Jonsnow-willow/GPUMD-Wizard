from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wizard.torchNEP.config import TrainConfig
from wizard.torchNEP.distributed import unwrap_model


class CheckpointManager:
    def __init__(self, config: TrainConfig, create_dirs: bool = True):
        self.config = config
        if create_dirs:
            self.config.ensure_output_dirs()

    def path(self, name: str) -> Path:
        return self.config.checkpoints_dir / name

    def save(
        self,
        name: str,
        model,
        optimizer,
        epoch: int,
        best_metric: float,
        metrics: dict[str, Any] | None = None,
        scheduler=None,
    ) -> Path:
        path = self.path(name)
        base_model = unwrap_model(model)
        torch.save(
            {
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "metrics": metrics or {},
                "para": base_model.para,
                "config": self.config.to_dict(),
                "rng_state": capture_rng_state(),
            },
            path,
        )
        return path

    def load(
        self,
        path: str | Path,
        model,
        optimizer=None,
        scheduler=None,
        device=None,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        path = Path(path)
        if not path.is_absolute():
            path = self.config.run_dir / path
        checkpoint = load_checkpoint_file(path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if restore_rng:
            restore_rng_state(checkpoint.get("rng_state"))
        return checkpoint


def load_checkpoint_file(path: str | Path, map_location=None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().cpu()
        torch.set_rng_state(torch_state)
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [
            cuda_state.detach().cpu() if isinstance(cuda_state, torch.Tensor) else cuda_state
            for cuda_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
