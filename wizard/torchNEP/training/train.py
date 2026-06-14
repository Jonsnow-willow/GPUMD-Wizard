from __future__ import annotations

from pathlib import Path

from wizard.torchNEP.config import TrainConfig
from wizard.torchNEP.parser import load_train_config
from wizard.torchNEP.training.trainer import GradientTrainer


def train_run(run_dir: str | Path):
    """Train a TorchNEP run directory containing nep.in and train.xyz."""
    return train_from_config(load_train_config(run_dir))


def train_from_config(config: TrainConfig):
    """Train from an already parsed TrainConfig."""
    return GradientTrainer(config).fit()
