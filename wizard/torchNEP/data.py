from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from wizard.torchNEP.config import TrainConfig
from wizard.torchNEP.dataset import StructureDataset, collate_fn
from wizard.torchNEP.distributed import DistributedContext
from wizard.torchNEP.lazy_xyz import LazyExtendedXYZDataset, default_index_path
from wizard.utils.io import read_xyz


def build_dataloaders(config: TrainConfig, context: DistributedContext | None = None):
    train_dataset = build_dataset(config, split="train", context=context)
    test_dataset = (
        build_dataset(config, split="test", context=context)
        if config.test_path and config.test_path.exists()
        else None
    )
    train_sampler = build_sampler(
        train_dataset,
        context=context,
        shuffle=config.data.shuffle,
        seed=config.runtime.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle if train_sampler is None else False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        **dataloader_performance_kwargs(config, context),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            sampler=build_sampler(
                test_dataset,
                context=context,
                shuffle=False,
                seed=config.runtime.seed,
            ),
            collate_fn=collate_fn,
            **dataloader_performance_kwargs(config, context),
        )
    return train_loader, test_loader


def dataloader_performance_kwargs(config: TrainConfig, context: DistributedContext | None = None) -> dict:
    num_workers = config.data.num_workers
    pin_memory_default = context is not None and context.device.type == "cuda"
    pin_memory = resolve_auto_bool(config.data.pin_memory, default=pin_memory_default)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = resolve_auto_bool(config.data.persistent_workers, default=True)
        if config.data.prefetch_factor is not None:
            if config.data.prefetch_factor < 1:
                raise ValueError("prefetch_factor must be >= 1.")
            kwargs["prefetch_factor"] = config.data.prefetch_factor
    return kwargs


def resolve_auto_bool(value: bool | str, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value == "auto":
        return default
    raise ValueError(f"Expected boolean or auto, got {value!r}.")


def build_sampler(dataset, context: DistributedContext | None, shuffle: bool, seed: int):
    if context is None or not context.is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )


def build_dataset(config: TrainConfig, split: str, context: DistributedContext | None = None):
    if config.data.frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")
    if split == "train":
        xyz_path = config.train_path
        max_frames = config.data.max_train_frames
    elif split == "test":
        xyz_path = config.test_path
        if xyz_path is None:
            raise ValueError("No test file configured.")
        max_frames = config.data.max_test_frames
    else:
        raise ValueError(f"Unknown split: {split}")

    if should_use_lazy_dataset(config, xyz_path):
        return build_lazy_dataset(config, split=split, xyz_path=xyz_path, max_frames=max_frames, context=context)

    frames = read_xyz(str(xyz_path))
    if config.data.frame_stride > 1:
        frames = frames[:: config.data.frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No frames found in {xyz_path}")
    return StructureDataset(frames=frames, para=config.model.to_nep_para(), require_forces=True)


def should_use_lazy_dataset(config: TrainConfig, xyz_path) -> bool:
    data_format = config.data.data_format
    if data_format == "eager":
        return False
    if data_format == "lazy_xyz":
        return True
    size_mb = xyz_path.stat().st_size / (1024 * 1024)
    return size_mb >= config.data.lazy_threshold_mb


def build_lazy_dataset(
    config: TrainConfig,
    split: str,
    xyz_path,
    max_frames: int | None,
    context: DistributedContext | None,
):
    index_path = default_index_path(
        config.run_dir,
        config.data.index_dir,
        xyz_path,
        split=split,
    )
    kwargs = {
        "xyz_path": xyz_path,
        "para": config.model.to_nep_para(),
        "index_path": index_path,
        "cache_index": config.data.cache_index,
        "frame_stride": config.data.frame_stride,
        "max_frames": max_frames,
        "require_forces": True,
    }
    if context is not None and context.is_distributed:
        dataset = None
        if context.is_main_process:
            dataset = LazyExtendedXYZDataset(**kwargs)
        context.barrier()
        if dataset is None:
            dataset = LazyExtendedXYZDataset(**kwargs)
        context.barrier()
        return dataset
    return LazyExtendedXYZDataset(**kwargs)
