from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from wizard.torchNEP.checkpoint import CheckpointManager
from wizard.torchNEP.config import TrainConfig
from wizard.torchNEP.data import build_dataloaders
from wizard.torchNEP.distributed import DistributedContext, unwrap_model
from wizard.torchNEP.metrics import MetricsLogger
from wizard.torchNEP.model import NEP


@dataclass
class EpochMetrics:
    total_loss: float = 0.0
    energy_loss: float = 0.0
    forces_loss: float = 0.0
    virial_loss: float = 0.0
    batches: int = 0
    seconds: float = 0.0

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_loss": self.total_loss,
            f"{prefix}_energy": self.energy_loss,
            f"{prefix}_forces": self.forces_loss,
            f"{prefix}_virial": self.virial_loss,
            f"{prefix}_seconds": self.seconds,
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
        self.loss_fn = nn.L1Loss()
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
        metrics = EpochMetrics()
        accumulation_steps = self.config.runtime.gradient_accumulation_steps
        if accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
        self.optimizer.zero_grad(set_to_none=True)
        total_steps = len(self.train_loader)
        final_accumulation_steps = total_steps % accumulation_steps or accumulation_steps
        for step, batch in enumerate(self.train_loader, 1):
            batch = self._prepare_batch(batch)
            should_step = step % accumulation_steps == 0 or step == total_steps
            scale = final_accumulation_steps if step == total_steps else accumulation_steps
            sync_context = nullcontext()
            if self.context.is_distributed and not should_step and hasattr(self.model, "no_sync"):
                sync_context = self.model.no_sync()
            with sync_context:
                prediction = self.model(batch)
                loss, loss_dict = self.compute_loss(prediction, batch)
                (loss / scale).backward()
            if should_step:
                self.clip_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            self._accumulate(metrics, loss.item(), loss_dict)
        metrics.seconds = time.perf_counter() - start
        return self._finalize_metrics(metrics)

    def evaluate(self) -> EpochMetrics:
        self.model.eval()
        start = time.perf_counter()
        metrics = EpochMetrics()
        for batch in self.test_loader:
            batch = self._prepare_batch(batch)
            prediction = self.model(batch)
            loss, loss_dict = self.compute_loss(prediction, batch)
            self._accumulate(metrics, float(loss.detach().cpu()), loss_dict)
        metrics.seconds = time.perf_counter() - start
        return self._finalize_metrics(metrics)

    def compute_loss(self, prediction, batch):
        weights = self.config.loss.weights()
        loss = None
        loss_dict = {}

        if "energy" in prediction and "energy" in batch and "is_energy" in batch:
            mask = batch["is_energy"]
            pred = prediction["energy"][mask]
            target = batch["energy"][mask]
            n_atoms_per_structure = batch["n_atoms_per_structure"][mask]
            pred = pred / n_atoms_per_structure.float()
            target = target / n_atoms_per_structure.float()
            energy_loss = self.loss_fn(pred, target)
            loss = _add_loss(loss, weights["energy"] * energy_loss)
            loss_dict["energy_loss"] = float(energy_loss.detach().cpu())

        if "forces" in prediction and "forces" in batch:
            forces_loss = self.loss_fn(prediction["forces"], batch["forces"])
            loss = _add_loss(loss, weights["forces"] * forces_loss)
            loss_dict["forces_loss"] = float(forces_loss.detach().cpu())

        if "virial" in prediction and "virial" in batch and "is_virial" in batch:
            mask = batch["is_virial"]
            pred = prediction["virial"][mask]
            target = batch["virial"][mask]
            n_atoms_per_structure = batch["n_atoms_per_structure"][mask]
            pred = pred / n_atoms_per_structure.float().unsqueeze(-1)
            target = target / n_atoms_per_structure.float().unsqueeze(-1)
            virial_loss = self.loss_fn(pred, target)
            loss = _add_loss(loss, weights["virial"] * virial_loss)
            loss_dict["virial_loss"] = float(virial_loss.detach().cpu())

        if loss is None:
            raise ValueError("No loss terms were available for this batch.")
        return loss, loss_dict

    def _prepare_batch(self, batch):
        prepared = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        prepared["compute_virial"] = "virial" in prepared and self.config.loss.virial != 0.0
        return prepared

    def _set_sampler_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _accumulate(self, metrics: EpochMetrics, total_loss: float, loss_dict: dict[str, float]) -> None:
        metrics.total_loss += total_loss
        metrics.energy_loss += loss_dict.get("energy_loss", 0.0)
        metrics.forces_loss += loss_dict.get("forces_loss", 0.0)
        metrics.virial_loss += loss_dict.get("virial_loss", 0.0)
        metrics.batches += 1

    def _finalize_metrics(self, metrics: EpochMetrics) -> EpochMetrics:
        if self.context.is_distributed:
            totals = torch.tensor(
                [
                    metrics.total_loss,
                    metrics.energy_loss,
                    metrics.forces_loss,
                    metrics.virial_loss,
                    float(metrics.batches),
                ],
                device=self.device,
                dtype=torch.float64,
            )
            self.context.all_reduce_sum(totals)
            metrics.total_loss = float(totals[0].item())
            metrics.energy_loss = float(totals[1].item())
            metrics.forces_loss = float(totals[2].item())
            metrics.virial_loss = float(totals[3].item())
            metrics.batches = int(totals[4].item())

            seconds = torch.tensor(metrics.seconds, device=self.device, dtype=torch.float64)
            self.context.all_reduce_max(seconds)
            metrics.seconds = float(seconds.item())
        return self._average(metrics)

    def _average(self, metrics: EpochMetrics) -> EpochMetrics:
        if metrics.batches == 0:
            return metrics
        metrics.total_loss /= metrics.batches
        metrics.energy_loss /= metrics.batches
        metrics.forces_loss /= metrics.batches
        metrics.virial_loss /= metrics.batches
        return metrics

    def _resume(self, resume_path: str) -> None:
        checkpoint = self.checkpoints.load(
            resume_path,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.best_metric = float(checkpoint.get("best_metric", float("inf")))
        self.q_scaler_initialized = True
        if self.is_main_process:
            print(f"Resumed from {resume_path} at epoch {self.current_epoch}.")

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
        message += f" | {train_metrics.seconds:.2f}s"
        print(message)


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


def _add_loss(current, term):
    return term if current is None else current + term
