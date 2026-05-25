from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    elements: list[str]
    rcut_radial: float = 8.0
    rcut_angular: float = 4.0
    n_max_radial: int = 4
    n_max_angular: int = 4
    basis_size_radial: int = 8
    basis_size_angular: int = 8
    l_max: int = 4
    l_max_4body: int = 2
    l_max_5body: int = 1
    nn_radial: int = 100
    nn_angular: int = 30
    hidden_dims: list[int] = field(default_factory=lambda: [30])
    zbl: dict[str, Any] | None = None

    def to_nep_para(self) -> dict[str, Any]:
        if not self.elements:
            raise ValueError("ModelConfig.elements must be provided explicitly.")
        para = {
            "elements": list(self.elements),
            "rcut_radial": self.rcut_radial,
            "rcut_angular": self.rcut_angular,
            "n_desc_radial": self.n_max_radial + 1,
            "n_desc_angular": self.n_max_angular + 1,
            "k_max_radial": self.basis_size_radial + 1,
            "k_max_angular": self.basis_size_angular + 1,
            "l_max": self.l_max,
            "l_max_4body": self.l_max_4body,
            "l_max_5body": self.l_max_5body,
            "NN_radial": self.nn_radial,
            "NN_angular": self.nn_angular,
            "hidden_dims": list(self.hidden_dims),
            "n_types": len(self.elements),
        }
        if self.zbl is not None:
            para["zbl"] = self.zbl
        return para


@dataclass
class DataConfig:
    train_file: str = "train.xyz"
    test_file: str | None = None
    batch_size: int = 5
    num_workers: int = 0
    shuffle: bool = True


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    epochs: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8

    @property
    def betas(self) -> tuple[float, float]:
        return self.beta1, self.beta2


@dataclass
class SchedulerConfig:
    name: str = "none"
    min_learning_rate: float = 0.0
    step_size: int = 100
    gamma: float = 0.5
    patience: int = 20
    factor: float = 0.5


@dataclass
class LossConfig:
    energy: float = 1.0
    forces: float = 1.0
    virial: float = 0.1

    def weights(self) -> dict[str, float]:
        return {"energy": self.energy, "forces": self.forces, "virial": self.virial}


@dataclass
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42
    save_every: int = 1
    export_every: int = 0
    resume: str | None = None
    compute_descriptor_scaler_once: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None


@dataclass
class TrainConfig:
    run_dir: Path
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def train_path(self) -> Path:
        return self._resolve_run_path(self.data.train_file)

    @property
    def test_path(self) -> Path | None:
        return self._resolve_run_path(self.data.test_file) if self.data.test_file else None

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def exports_dir(self) -> Path:
        return self.run_dir / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def ensure_output_dirs(self) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        return data

    def _resolve_run_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.run_dir / path
