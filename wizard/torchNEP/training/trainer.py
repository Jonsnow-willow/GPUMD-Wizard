from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from wizard.torchNEP.config import TrainConfig
from wizard.torchNEP.datasets.data import build_dataloaders
from wizard.torchNEP.nep.model import NEP
from wizard.torchNEP.quantities import independent_virial_components
from wizard.torchNEP.runtime.distributed import DistributedContext, unwrap_model
from wizard.torchNEP.training.checkpoint import CheckpointManager
from wizard.torchNEP.training.metrics import MetricsLogger


LOSS_TERMS = ("energy", "forces", "virial")


@dataclass
class EpochMetrics:
    total_loss: float = 0.0
    energy_loss: float = 0.0
    forces_loss: float = 0.0
    virial_loss: float = 0.0
    batches: int = 0
    seconds: float = 0.0
    data_seconds: float = 0.0
    compute_seconds: float = 0.0

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_loss": self.total_loss,
            f"{prefix}_energy": self.energy_loss,
            f"{prefix}_forces": self.forces_loss,
            f"{prefix}_virial": self.virial_loss,
            f"{prefix}_seconds": self.seconds,
            f"{prefix}_data_seconds": self.data_seconds,
            f"{prefix}_compute_seconds": self.compute_seconds,
        }


class GradientTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.context = DistributedContext.from_environment(config.runtime.device)

        set_seed(config.runtime.seed)
        self.device = self.context.device
        if self.context.is_main_process:
            self.config.ensure_output_dirs()
        self.context.barrier()

        self.model = self.context.wrap_model(NEP(config.model.to_nep_para()).to(self.device))
        self.train_loader, self.test_loader = build_dataloaders(config, context=self.context)
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.scheduler = build_scheduler(self.optimizer, config.optimizer, config.scheduler)
        self.checkpoints = CheckpointManager(config, create_dirs=self.context.is_main_process)
        self.metrics_logger = MetricsLogger(config.logs_dir / "metrics.csv") if self.context.is_main_process else None
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.q_scaler_initialized = False

        if config.runtime.resume:
            self._resume(config.runtime.resume)

    @property
    def is_main_process(self) -> bool:
        return self.context.is_main_process

    def fit(self) -> NEP:
        if self.config.optimizer.name != "adamw":
            raise ValueError(f"GradientTrainer currently supports optimizer adamw, got {self.config.optimizer.name}.")

        self._log_start()
        self.initialize_descriptor_scaler()

        last_row = None
        for epoch in range(self.current_epoch + 1, self.config.optimizer.epochs + 1):
            self.current_epoch = epoch
            self._set_sampler_epoch(epoch)
            train_metrics = self.train_epoch()
            test_metrics = self.evaluate() if self.test_loader is not None else None
            score = test_metrics.total_loss if test_metrics is not None else train_metrics.total_loss
            self.step_scheduler(score)

            row = {"epoch": epoch, "lr": self.current_learning_rate(), **train_metrics.as_dict("train")}
            if test_metrics is not None:
                row.update(test_metrics.as_dict("test"))
            last_row = row

            if self.is_main_process:
                self.metrics_logger.log(row)
                self._log_epoch(epoch, train_metrics, test_metrics)
                if score < self.best_metric:
                    self.best_metric = score
                    self.checkpoints.save(
                        "best.pt",
                        self.model,
                        self.optimizer,
                        epoch,
                        self.best_metric,
                        row,
                        scheduler=self.scheduler,
                    )
                if self.config.runtime.save_every > 0 and epoch % self.config.runtime.save_every == 0:
                    self.checkpoints.save(
                        "last.pt",
                        self.model,
                        self.optimizer,
                        epoch,
                        self.best_metric,
                        row,
                        scheduler=self.scheduler,
                    )
                if self.config.runtime.export_every > 0 and epoch % self.config.runtime.export_every == 0:
                    unwrap_model(self.model).save_to_nep_format(self.config.exports_dir / f"nep_epoch_{epoch}.txt")

        if self.is_main_process:
            self.checkpoints.save(
                "last.pt",
                self.model,
                self.optimizer,
                self.current_epoch,
                self.best_metric,
                last_row,
                scheduler=self.scheduler,
            )
            unwrap_model(self.model).save_to_nep_format(self.config.exports_dir / "nep.txt")
            print("Training completed.")
        self.context.barrier()
        model = unwrap_model(self.model)
        self.context.close()
        return model

    def initialize_descriptor_scaler(self) -> None:
        if not self.config.runtime.compute_descriptor_scaler_once or self.q_scaler_initialized:
            return
        if self.current_epoch > 0:
            self.q_scaler_initialized = True
            return
        base_model = unwrap_model(self.model)
        max_batches = self.config.runtime.descriptor_scaler_max_batches
        if max_batches is not None and max_batches < 1:
            raise ValueError("descriptor_scaler_max_batches must be >= 1.")
        if self.context.is_distributed:
            q_min, q_max = base_model.compute_descriptor_min_max(
                self.train_loader,
                device=self.device,
                max_batches=max_batches,
            )
            if q_min is None or q_max is None:
                q_min = torch.full_like(base_model.q_scaler, float("inf"))
                q_max = torch.full_like(base_model.q_scaler, float("-inf"))
            self.context.all_reduce_min(q_min)
            self.context.all_reduce_max(q_max)
            scaler = base_model.set_descriptor_scaler_from_min_max(q_min, q_max)
        else:
            scaler = base_model.compute_descriptor_scaler(
                self.train_loader,
                device=self.device,
                max_batches=max_batches,
            )
        self.q_scaler_initialized = True
        if self.is_main_process:
            sample_text = "" if max_batches is None else f" from at most {max_batches} batch(es)"
            print(
                f"Initialized q_scaler once{sample_text} "
                f"(min={torch.min(scaler).item():.6e}, max={torch.max(scaler).item():.6e})."
            )

    def train_epoch(self) -> EpochMetrics:
        self.model.train()
        start = time.perf_counter()
        loss_sums = torch.zeros(3, device=self.device, dtype=torch.float64)
        loss_counts = torch.zeros(3, device=self.device, dtype=torch.float64)
        batches = 0
        data_seconds = 0.0
        compute_seconds = 0.0
        accumulation_steps = self.config.runtime.gradient_accumulation_steps
        if accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
        self.optimizer.zero_grad(set_to_none=True)
        total_steps = len(self.train_loader)
        iterator = iter(self.train_loader)

        for window_start in range(0, total_steps, accumulation_steps):
            window_size = min(accumulation_steps, total_steps - window_start)
            data_start = time.perf_counter()
            window_batches = [next(iterator) for _ in range(window_size)]
            data_seconds += time.perf_counter() - data_start

            count_start = time.perf_counter()
            normalization_counts = self._global_loss_counts(window_batches)
            compute_seconds += time.perf_counter() - count_start

            for window_index, batch in enumerate(window_batches):
                step = window_start + window_index + 1
                compute_start = time.perf_counter()
                batch = self._prepare_batch(batch)
                should_step = window_index == window_size - 1
                sync_context = nullcontext()
                if self.context.is_distributed and not should_step and hasattr(self.model, "no_sync"):
                    sync_context = self.model.no_sync()
                with sync_context:
                    prediction = self.model(batch)
                    loss, loss_statistics = self.compute_loss(
                        prediction,
                        batch,
                        normalization_counts=normalization_counts,
                    )
                    loss.backward()
                if should_step:
                    self.clip_gradients()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                self._accumulate_loss_statistics(loss_sums, loss_counts, loss_statistics)
                batches += 1
                compute_seconds += time.perf_counter() - compute_start
                self._log_progress(
                    epoch=self.current_epoch,
                    step=step,
                    total_steps=total_steps,
                    loss_sums=loss_sums,
                    loss_counts=loss_counts,
                    batches=batches,
                    seconds=time.perf_counter() - start,
                    data_seconds=data_seconds,
                    compute_seconds=compute_seconds,
                )
        return self._finalize_metrics(
            loss_sums=loss_sums,
            loss_counts=loss_counts,
            batches=batches,
            seconds=time.perf_counter() - start,
            data_seconds=data_seconds,
            compute_seconds=compute_seconds,
        )

    def evaluate(self) -> EpochMetrics:
        self.model.eval()
        start = time.perf_counter()
        loss_sums = torch.zeros(3, device=self.device, dtype=torch.float64)
        loss_counts = torch.zeros(3, device=self.device, dtype=torch.float64)
        batches = 0
        data_seconds = 0.0
        compute_seconds = 0.0
        iterator = iter(self.test_loader)
        total_steps = len(self.test_loader)
        for _ in range(total_steps):
            data_start = time.perf_counter()
            batch = next(iterator)
            data_seconds += time.perf_counter() - data_start
            if not any(_batch_loss_counts(batch)):
                raise ValueError("No loss terms were available for this batch.")
            compute_start = time.perf_counter()
            batch = self._prepare_batch(batch)
            prediction = self.model(batch)
            loss_statistics = self._loss_statistics(prediction, batch)
            self._accumulate_loss_statistics(loss_sums, loss_counts, loss_statistics)
            batches += 1
            compute_seconds += time.perf_counter() - compute_start
        return self._finalize_metrics(
            loss_sums=loss_sums,
            loss_counts=loss_counts,
            batches=batches,
            seconds=time.perf_counter() - start,
            data_seconds=data_seconds,
            compute_seconds=compute_seconds,
        )

    def compute_loss(self, prediction, batch, normalization_counts: torch.Tensor | None = None):
        weights = self.config.loss.weights()
        statistics = self._loss_statistics(prediction, batch)

        if normalization_counts is None:
            normalization_counts = statistics["counts"].detach().clone()
            self.context.all_reduce_sum(normalization_counts)
            if not torch.any(normalization_counts > 0).item():
                raise ValueError("No loss terms were available for this batch.")
        if normalization_counts.shape != statistics["counts"].shape:
            raise ValueError(
                f"Expected {len(LOSS_TERMS)} loss counts, got shape {tuple(normalization_counts.shape)}."
            )

        # DDP averages gradients across ranks. Multiplying each local numerator by
        # world_size therefore produces a true global sum/global-count gradient.
        ddp_scale = float(self.context.world_size if self.context.is_distributed else 1)
        safe_counts = normalization_counts.to(
            device=statistics["sums"].device,
            dtype=statistics["sums"].dtype,
        ).clamp_min(1.0)
        active_terms = (normalization_counts > 0).to(
            device=statistics["sums"].device,
            dtype=statistics["sums"].dtype,
        )
        loss_weights = statistics["sums"].new_tensor([weights[name] for name in LOSS_TERMS])
        normalized_terms = statistics["sums"] * ddp_scale / safe_counts * active_terms
        loss = torch.sum(loss_weights * normalized_terms)
        return loss, statistics

    def _loss_statistics(self, prediction, batch) -> dict[str, torch.Tensor]:
        reference = next(
            (value for value in prediction.values() if isinstance(value, torch.Tensor)),
            None,
        )
        if reference is None:
            raise ValueError("Model prediction did not contain any tensors.")
        zero = reference.sum() * 0.0
        sums = [zero, zero, zero]
        counts = [0, 0, 0]

        if "energy" in prediction and "energy" in batch and "is_energy" in batch:
            mask = batch["is_energy"]
            pred = prediction["energy"][mask]
            target = batch["energy"][mask]
            n_atoms_per_structure = batch["n_atoms_per_structure"][mask]
            pred = pred / n_atoms_per_structure.float()
            target = target / n_atoms_per_structure.float()
            residual = torch.abs(pred - target)
            sums[0] = residual.sum()
            counts[0] = residual.numel()

        if "forces" in prediction and "forces" in batch:
            residual = torch.abs(prediction["forces"] - batch["forces"])
            sums[1] = residual.sum()
            counts[1] = residual.numel()

        if "virial" in prediction and "virial" in batch and "is_virial" in batch:
            mask = batch["is_virial"]
            pred = independent_virial_components(prediction["virial"][mask])
            target = independent_virial_components(batch["virial"][mask])
            n_atoms_per_structure = batch["n_atoms_per_structure"][mask]
            pred = pred / n_atoms_per_structure.float().unsqueeze(-1)
            target = target / n_atoms_per_structure.float().unsqueeze(-1)
            residual = torch.abs(pred - target)
            sums[2] = residual.sum()
            counts[2] = residual.numel()

        return {
            "sums": torch.stack(sums),
            "counts": reference.new_tensor(counts),
        }

    def _global_loss_counts(self, batches) -> torch.Tensor:
        local_counts = [0, 0, 0]
        for batch in batches:
            batch_counts = _batch_loss_counts(batch)
            for index, count in enumerate(batch_counts):
                local_counts[index] += count
        counts = torch.tensor(local_counts, device=self.device, dtype=torch.float32)
        self.context.all_reduce_sum(counts)
        if not torch.any(counts > 0).item():
            raise ValueError("No loss terms were available for this accumulation window.")
        return counts

    def _prepare_batch(self, batch):
        non_blocking = self.device.type == "cuda"
        prepared = {
            key: value.to(self.device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        prepared["compute_virial"] = "virial" in prepared and self.config.loss.virial != 0.0
        return prepared

    def _set_sampler_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _accumulate_loss_statistics(
        self,
        loss_sums: torch.Tensor,
        loss_counts: torch.Tensor,
        statistics: dict[str, torch.Tensor],
    ) -> None:
        loss_sums += statistics["sums"].detach().to(dtype=torch.float64)
        loss_counts += statistics["counts"].detach().to(dtype=torch.float64)

    def _finalize_metrics(
        self,
        loss_sums: torch.Tensor,
        loss_counts: torch.Tensor,
        batches: int,
        seconds: float,
        data_seconds: float,
        compute_seconds: float,
    ) -> EpochMetrics:
        if self.context.is_distributed:
            totals = torch.cat([
                loss_sums,
                loss_counts,
                torch.tensor([float(batches)], device=self.device, dtype=torch.float64),
            ])
            self.context.all_reduce_sum(totals)
            loss_sums = totals[:3]
            loss_counts = totals[3:6]
            batches = int(totals[6].item())

            seconds = torch.tensor(
                [seconds, data_seconds, compute_seconds],
                device=self.device,
                dtype=torch.float64,
            )
            self.context.all_reduce_max(seconds)
            seconds, data_seconds, compute_seconds = (
                float(seconds[0].item()),
                float(seconds[1].item()),
                float(seconds[2].item()),
            )
        else:
            loss_sums = loss_sums.detach()
            loss_counts = loss_counts.detach()

        mean_losses = torch.where(
            loss_counts > 0,
            loss_sums / loss_counts.clamp_min(1.0),
            torch.zeros_like(loss_sums),
        )
        weights = self.config.loss.weights()
        total_loss = sum(weights[name] * float(mean_losses[index].item()) for index, name in enumerate(LOSS_TERMS))

        metrics = EpochMetrics(
            total_loss=total_loss,
            energy_loss=float(mean_losses[0].item()),
            forces_loss=float(mean_losses[1].item()),
            virial_loss=float(mean_losses[2].item()),
            batches=batches,
            seconds=seconds,
            data_seconds=data_seconds,
            compute_seconds=compute_seconds,
        )
        return metrics

    def _log_progress(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        loss_sums: torch.Tensor,
        loss_counts: torch.Tensor,
        batches: int,
        seconds: float,
        data_seconds: float,
        compute_seconds: float,
    ) -> None:
        interval = self.config.runtime.progress_log_interval
        if interval <= 0 or step == total_steps or step % interval != 0:
            return
        metrics = self._finalize_metrics(
            loss_sums=loss_sums.clone(),
            loss_counts=loss_counts.clone(),
            batches=batches,
            seconds=seconds,
            data_seconds=data_seconds,
            compute_seconds=compute_seconds,
        )
        if not self.is_main_process:
            return
        steps_per_second = step / max(metrics.seconds, 1.0e-12)
        print(
            f"Epoch {epoch:4d}/{self.config.optimizer.epochs} "
            f"step {step:6d}/{total_steps}: "
            f"train={metrics.total_loss:.6f} "
            f"E={metrics.energy_loss:.6f} "
            f"F={metrics.forces_loss:.6f} "
            f"V={metrics.virial_loss:.6f} "
            f"lr={self.current_learning_rate():.6g} "
            f"| {metrics.seconds:.2f}s "
            f"(data={metrics.data_seconds:.2f}s compute={metrics.compute_seconds:.2f}s "
            f"{steps_per_second:.4f} step/s)",
            flush=True,
        )

    def _resume(self, resume_path: str) -> None:
        optimizer = None if self.config.runtime.resume_model_only else self.optimizer
        scheduler = None if self.config.runtime.resume_model_only else self.scheduler
        checkpoint = self.checkpoints.load(
            resume_path,
            self.model,
            optimizer,
            scheduler=scheduler,
            device=self.device,
            restore_rng=not self.config.runtime.resume_model_only,
        )
        self.current_epoch = int(checkpoint.get("epoch", 0))
        if self.config.runtime.resume_model_only or self.config.runtime.resume_reset_best:
            self.best_metric = float("inf")
        else:
            self.best_metric = float(checkpoint.get("best_metric", float("inf")))
        self.q_scaler_initialized = True
        if self.is_main_process:
            mode = "model only" if self.config.runtime.resume_model_only else "full state"
            print(f"Resumed {mode} from {resume_path} at epoch {self.current_epoch}.", flush=True)

    def step_scheduler(self, score: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def clip_gradients(self) -> None:
        clip_norm = self.config.runtime.gradient_clip_norm
        if clip_norm is None or clip_norm <= 0:
            return
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

    def _log_start(self) -> None:
        if not self.is_main_process:
            return
        print(f"Run directory: {self.config.run_dir}")
        print(f"Training data: {self.config.train_path}")
        if self.config.test_path:
            print(f"Test data: {self.config.test_path}")
        print(f"Elements: {self.config.model.elements}")
        print(
            f"Optimizer: {self.config.optimizer.name} "
            f"lr={self.config.optimizer.learning_rate} "
            f"weight_decay={self.config.optimizer.weight_decay}"
        )
        print(f"Scheduler: {self.config.scheduler.name}")
        print(f"Device: {self.device}")
        print(f"Execution: {self.context.describe()}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters())}")

    def _log_epoch(self, epoch: int, train_metrics: EpochMetrics, test_metrics: EpochMetrics | None) -> None:
        message = (
            f"Epoch {epoch:4d}/{self.config.optimizer.epochs}: "
            f"train={train_metrics.total_loss:.6f} "
            f"E={train_metrics.energy_loss:.6f} "
            f"F={train_metrics.forces_loss:.6f} "
            f"V={train_metrics.virial_loss:.6f}"
        )
        if test_metrics is not None:
            message += f" | test={test_metrics.total_loss:.6f}"
        message += (
            f" | {train_metrics.seconds:.2f}s "
            f"(data={train_metrics.data_seconds:.2f}s compute={train_metrics.compute_seconds:.2f}s)"
        )
        print(message, flush=True)


def build_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    if config.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {config.name}. Use optimizer adamw.")

    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    parameter_groups = []
    if decay_params:
        parameter_groups.append({"params": decay_params, "weight_decay": config.weight_decay})
    if no_decay_params:
        parameter_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, optimizer_config, scheduler_config):
    name = scheduler_config.name
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, optimizer_config.epochs),
            eta_min=scheduler_config.min_learning_rate,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, scheduler_config.step_size),
            gamma=scheduler_config.gamma,
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.min_learning_rate,
        )
    raise ValueError(f"Unsupported scheduler: {name}. Use none, cosine, step, or plateau.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_loss_counts(batch) -> tuple[int, int, int]:
    energy_count = 0
    if "energy" in batch and "is_energy" in batch:
        energy_count = int(torch.count_nonzero(batch["is_energy"]).item())

    forces_count = int(batch["forces"].numel()) if "forces" in batch else 0

    virial_count = 0
    if "virial" in batch and "is_virial" in batch:
        virial = batch["virial"][batch["is_virial"]]
        virial_count = independent_virial_components(virial).numel()
    return energy_count, forces_count, virial_count
