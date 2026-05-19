from __future__ import annotations

from pathlib import Path
from typing import Any

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
    ) -> Path:
        path = self.path(name)
        base_model = unwrap_model(model)
        torch.save(
            {
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "epoch": epoch,
                "best_metric": best_metric,
                "metrics": metrics or {},
                "para": base_model.para,
                "config": self.config.to_dict(),
            },
            path,
        )
        return path

    def load(self, path: str | Path, model, optimizer=None, device=None) -> dict[str, Any]:
        path = Path(path)
        if not path.is_absolute():
            path = self.config.run_dir / path
        checkpoint = torch.load(path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
