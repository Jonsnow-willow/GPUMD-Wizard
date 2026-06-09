from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device

    @classmethod
    def from_environment(cls, device: str | torch.device | None = "auto") -> "DistributedContext":
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        resolved_device = resolve_device(device, local_rank=local_rank)
        backend = _select_backend(resolved_device)
        context = cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            device=resolved_device,
        )
        context.initialize_process_group()
        return context

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def process_group_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def initialize_process_group(self) -> None:
        if not self.is_distributed:
            return
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this PyTorch build.")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if not dist.is_initialized():
            try:
                if self.device.type == "cuda":
                    dist.init_process_group(backend=self.backend, device_id=self.device)
                else:
                    dist.init_process_group(backend=self.backend)
            except TypeError:
                dist.init_process_group(backend=self.backend)

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.is_distributed:
            return model
        if self.device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=True,
            )
        return DistributedDataParallel(model, find_unused_parameters=True)

    def barrier(self) -> None:
        if self.process_group_initialized:
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.device.index])
            else:
                dist.barrier()

    def broadcast_tensor(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.process_group_initialized:
            dist.broadcast(tensor, src=src)
        return tensor

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.process_group_initialized:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.process_group_initialized:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return tensor

    def all_reduce_max(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.process_group_initialized:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor

    def close(self) -> None:
        if self.process_group_initialized:
            dist.destroy_process_group()

    def describe(self) -> str:
        if not self.is_distributed:
            return "single-process"
        return (
            f"distributed rank={self.rank} local_rank={self.local_rank} "
            f"world_size={self.world_size} backend={self.backend}"
        )


def resolve_device(device: str | torch.device | None, local_rank: int = 0) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    requested = str(device).lower()
    if requested == "gpu":
        requested = "cuda"
    if requested == "cuda":
        requested = f"cuda:{local_rank}"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False in this environment.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False in this environment.")
    return torch.device(requested)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _select_backend(device: torch.device) -> str:
    return "nccl" if device.type == "cuda" else "gloo"
