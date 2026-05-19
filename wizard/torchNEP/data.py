from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from wizard.torchNEP.config import TrainConfig
from wizard.torchNEP.dataset import StructureDataset, collate_fn
from wizard.torchNEP.distributed import DistributedContext
from wizard.utils.io import read_xyz


def build_dataloaders(config: TrainConfig, context: DistributedContext | None = None):
    train_dataset = build_dataset(config, split="train")
    test_dataset = build_dataset(config, split="test") if config.test_path and config.test_path.exists() else None
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
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
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
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
        )
    return train_loader, test_loader


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


def build_dataset(config: TrainConfig, split: str):
    if split == "train":
        xyz_path = config.train_path
    elif split == "test":
        xyz_path = config.test_path
        if xyz_path is None:
            raise ValueError("No test file configured.")
    else:
        raise ValueError(f"Unknown split: {split}")

    frames = read_xyz(str(xyz_path))
    if not frames:
        raise ValueError(f"No frames found in {xyz_path}")
    return StructureDataset(frames=frames, para=config.model.to_nep_para(), require_forces=True)
