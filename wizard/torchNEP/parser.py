from __future__ import annotations

from pathlib import Path

from wizard.torchNEP.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SchedulerConfig,
    TrainConfig,
)


def load_train_config(run_dir: str | Path) -> TrainConfig:
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "nep.in"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    parsed = _parse_key_value_file(config_path)
    model = _build_model_config(parsed, run_dir)
    data = _build_data_config(parsed)
    optimizer = _build_optimizer_config(parsed)
    scheduler = _build_scheduler_config(parsed)
    loss = _build_loss_config(parsed)
    runtime = _build_runtime_config(parsed)
    return TrainConfig(
        run_dir=run_dir,
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        runtime=runtime,
    )


def _parse_key_value_file(path: Path) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    with path.open() as f:
        for line_number, raw_line in enumerate(f, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            key = tokens[0].lower()
            values = tokens[1:]
            if key in parsed:
                raise ValueError(f"Duplicate key '{key}' in {path} line {line_number}.")
            parsed[key] = values
    return parsed


def _build_model_config(parsed: dict[str, list[str]], run_dir: Path) -> ModelConfig:
    version = _pop_scalar(parsed, "version", default="nep4").lower()
    if version == "4":
        version = "nep4"
    if version != "nep4":
        raise NotImplementedError("TorchNEP training currently supports only NEP4.")

    elements = parsed.pop("elements", None)
    if elements is None:
        elements = _parse_type_line(parsed.pop("type", None))
    if not elements:
        raise ValueError("nep.in must define `elements Te Pb` or GPUMD-style `type 2 Te Pb`.")

    cutoff = _pop_required(parsed, "cutoff")
    if len(cutoff) == 2:
        neighbor_counts = _pop_neighbor_counts(parsed)
        cutoff = [*cutoff, *neighbor_counts]
    elif len(cutoff) != 4:
        raise NotImplementedError("TorchNEP currently supports global cutoff: cutoff rc_radial rc_angular [NN_radial NN_angular]")

    n_max = _pop_required(parsed, "n_max")
    basis_size = _pop_required(parsed, "basis_size")
    l_max = _pop_required(parsed, "l_max")
    _require_length("n_max", n_max, 2)
    _require_length("basis_size", basis_size, 2)
    _require_length("l_max", l_max, 3)
    neuron = parsed.pop("neuron", None)
    if neuron is None:
        neuron = parsed.pop("ann", ["30"])
    if len(neuron) == 2 and neuron[1] == "0":
        neuron = [neuron[0]]

    zbl = _parse_zbl(parsed, run_dir, len(elements))

    return ModelConfig(
        elements=elements,
        rcut_radial=float(cutoff[0]),
        rcut_angular=float(cutoff[1]),
        nn_radial=int(cutoff[2]),
        nn_angular=int(cutoff[3]),
        n_max_radial=int(n_max[0]),
        n_max_angular=int(n_max[1]),
        basis_size_radial=int(basis_size[0]),
        basis_size_angular=int(basis_size[1]),
        l_max=int(l_max[0]),
        l_max_4body=int(l_max[1]),
        l_max_5body=int(l_max[2]),
        hidden_dims=[int(value) for value in neuron],
        zbl=zbl,
    )


def _parse_zbl(parsed: dict[str, list[str]], run_dir: Path, n_types: int) -> dict | None:
    zbl_tokens = parsed.pop("zbl", None)
    if zbl_tokens is None:
        return None
    if len(zbl_tokens) not in (2, 3):
        raise ValueError("zbl should be: zbl rc_inner rc_outer [zbl_factor]")
    rc_inner = float(zbl_tokens[0])
    rc_outer = float(zbl_tokens[1])
    if len(zbl_tokens) == 3:
        raise NotImplementedError("Typewise ZBL cutoffs are not supported in TorchNEP training.")
    zbl = {
        "rc_inner": rc_inner,
        "rc_outer": rc_outer,
        "flexible": rc_inner == 0.0 and rc_outer == 0.0,
    }
    if zbl["flexible"]:
        zbl_file = _pop_scalar(parsed, "zbl_file", default="zbl.in")
        zbl["parameters"] = _read_zbl_parameters(run_dir / zbl_file, n_types)
    return zbl


def _read_zbl_parameters(path: Path, n_types: int) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Flexible ZBL requested, but zbl.in was not found: {path}")
    values: list[float] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                values.extend(float(token) for token in line.split())
    expected = 10 * n_types * (n_types + 1) // 2
    if len(values) != expected:
        raise ValueError(f"{path} should contain {expected} flexible ZBL parameters, found {len(values)}.")
    return values


def _build_data_config(parsed: dict[str, list[str]]) -> DataConfig:
    train_file = _pop_scalar(parsed, "train", default="train.xyz")
    test_file = _pop_scalar(parsed, "test", default=None)
    batch_size = _pop_scalar(parsed, "batch_size", default=None)
    if batch_size is None:
        batch_size = _pop_scalar(parsed, "batch", default="5")
    return DataConfig(
        train_file=train_file,
        test_file=test_file,
        data_format=_normalize_data_format(_pop_scalar(parsed, "data_format", default="auto")),
        batch_size=int(batch_size),
        num_workers=int(_pop_scalar(parsed, "num_workers", default="0")),
        pin_memory=_parse_auto_bool(_pop_scalar(parsed, "pin_memory", default="auto")),
        persistent_workers=_parse_auto_bool(_pop_scalar(parsed, "persistent_workers", default="auto")),
        prefetch_factor=_parse_optional_int(_pop_scalar(parsed, "prefetch_factor", default="2")),
        shuffle=_parse_bool(_pop_scalar(parsed, "shuffle", default="true")),
        lazy_threshold_mb=float(_pop_scalar(parsed, "lazy_threshold_mb", default="512.0")),
        max_train_frames=_parse_optional_int(_pop_scalar(parsed, "max_train_frames", default=None)),
        max_test_frames=_parse_optional_int(_pop_scalar(parsed, "max_test_frames", default=None)),
        frame_stride=int(_pop_scalar(parsed, "frame_stride", default="1")),
        index_dir=_pop_scalar(parsed, "index_dir", default=".torchnep_index"),
        cache_index=_parse_bool(_pop_scalar(parsed, "cache_index", default="true")),
    )


def _build_optimizer_config(parsed: dict[str, list[str]]) -> OptimizerConfig:
    learning_rate = _pop_scalar(parsed, "learning_rate", default=None)
    if learning_rate is None:
        learning_rate = _pop_scalar(parsed, "lr", default="1.0e-3")
    epochs = _pop_scalar(parsed, "epochs", default=None)
    if epochs is None:
        epochs = _pop_scalar(parsed, "epoch", default="500")

    betas = parsed.pop("betas", None)
    if betas is not None:
        _require_length("betas", betas, 2)
        beta1, beta2 = betas
    else:
        beta1 = _pop_scalar(parsed, "beta1", default="0.9")
        beta2 = _pop_scalar(parsed, "beta2", default="0.999")

    return OptimizerConfig(
        name=_normalize_optimizer_name(_pop_scalar(parsed, "optimizer", default="adamw")),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        weight_decay=float(_pop_first_scalar(parsed, ("weight_decay", "wd"), default="1.0e-4")),
        beta1=float(beta1),
        beta2=float(beta2),
        eps=float(_pop_scalar(parsed, "eps", default="1.0e-8")),
    )


def _build_scheduler_config(parsed: dict[str, list[str]]) -> SchedulerConfig:
    return SchedulerConfig(
        name=_normalize_scheduler_name(_pop_scalar(parsed, "scheduler", default="none")),
        min_learning_rate=float(_pop_first_scalar(parsed, ("min_learning_rate", "min_lr"), default="0.0")),
        step_size=int(_pop_first_scalar(parsed, ("lr_step_size", "scheduler_step_size"), default="100")),
        gamma=float(_pop_first_scalar(parsed, ("lr_gamma", "scheduler_gamma"), default="0.5")),
        patience=int(_pop_first_scalar(parsed, ("lr_patience", "scheduler_patience"), default="20")),
        factor=float(_pop_first_scalar(parsed, ("lr_factor", "scheduler_factor"), default="0.5")),
    )


def _build_loss_config(parsed: dict[str, list[str]]) -> LossConfig:
    lambda_energy = _pop_first_scalar(parsed, ("lambda_energy", "lambda_e"), default="1.0")
    lambda_force = _pop_first_scalar(parsed, ("lambda_force", "lambda_forces", "lambda_f"), default="1.0")
    lambda_virial = _pop_first_scalar(parsed, ("lambda_virial", "lambda_v"), default="0.1")
    return LossConfig(
        energy=float(lambda_energy),
        forces=float(lambda_force),
        virial=float(lambda_virial),
    )


def _build_runtime_config(parsed: dict[str, list[str]]) -> RuntimeConfig:
    runtime = RuntimeConfig(
        device=_pop_scalar(parsed, "device", default="auto"),
        seed=int(_pop_scalar(parsed, "seed", default="42")),
        save_every=int(_pop_scalar(parsed, "save_every", default="1")),
        export_every=int(_pop_scalar(parsed, "export_every", default="0")),
        resume=_pop_scalar(parsed, "resume", default=None),
        progress_log_interval=int(
            _pop_first_scalar(parsed, ("progress_log_interval", "log_interval"), default="0")
        ),
        compute_descriptor_scaler_once=_parse_bool(_pop_scalar(parsed, "compute_descriptor_scaler_once", default="true")),
        gradient_accumulation_steps=int(
            _pop_first_scalar(parsed, ("gradient_accumulation_steps", "accumulation_steps"), default="1")
        ),
        gradient_clip_norm=_parse_optional_float(
            _pop_first_scalar(parsed, ("gradient_clip_norm", "clip_grad_norm"), default=None)
        ),
        descriptor_scaler_max_batches=_parse_optional_int(
            _pop_first_scalar(parsed, ("descriptor_scaler_max_batches", "q_scaler_max_batches"), default=None)
        ),
    )
    if parsed:
        raise ValueError(f"Unsupported nep.in keys: {sorted(parsed)}")
    return runtime


def _pop_required(parsed: dict[str, list[str]], key: str) -> list[str]:
    values = parsed.pop(key, None)
    if values is None:
        raise ValueError(f"nep.in must define `{key}`.")
    return values


def _pop_scalar(parsed: dict[str, list[str]], key: str, default=None):
    values = parsed.pop(key, None)
    if values is None:
        return default
    if len(values) != 1:
        raise ValueError(f"`{key}` expects exactly one value.")
    return values[0]


def _parse_type_line(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    if not values:
        raise ValueError("`type` expects: type <n_types> <element...>.")
    n_types = int(values[0])
    elements = values[1:]
    if len(elements) != n_types:
        raise ValueError(f"`type` declares {n_types} types but lists {len(elements)} elements.")
    return elements


def _pop_neighbor_counts(parsed: dict[str, list[str]]) -> list[str]:
    for key in ("max_neighbors", "num_neighbors", "neighbors"):
        values = parsed.pop(key, None)
        if values is not None:
            _require_length(key, values, 2)
            return values
    return ["100", "30"]


def _pop_first_scalar(parsed: dict[str, list[str]], keys: tuple[str, ...], default=None):
    for key in keys:
        if key in parsed:
            return _pop_scalar(parsed, key)
    return default


def _normalize_optimizer_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def _normalize_scheduler_name(name: str) -> str:
    normalized = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "off": "none",
        "null": "none",
        "cosineannealinglr": "cosine",
        "steplr": "step",
        "reducelronplateau": "plateau",
    }
    return aliases.get(normalized, normalized)


def _normalize_data_format(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    aliases = {
        "lazy": "lazy_xyz",
        "lazyxyz": "lazy_xyz",
        "extxyz": "lazy_xyz",
        "extended_xyz": "lazy_xyz",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "eager", "lazy_xyz"}:
        raise ValueError("data_format must be auto, eager, or lazy_xyz.")
    return normalized


def _require_length(key: str, values: list[str], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"`{key}` expects {expected} values, got {len(values)}.")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_auto_bool(value: str | bool) -> bool | str:
    if isinstance(value, bool):
        return value
    if value.lower() == "auto":
        return "auto"
    return _parse_bool(value)


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)
