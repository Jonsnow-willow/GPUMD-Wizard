from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wizard.io import read_xyz
from wizard.torchNEP.dataset import StructureDataset, collate_fn
from wizard.torchNEP.model import NEP
from wizard.torchNEP.optimizer import Optimizer, SNES


DEFAULT_MODEL_PARA = {
    "rcut_radial": 8.0,
    "rcut_angular": 4.0,
    "n_desc_radial": 5,
    "n_desc_angular": 5,
    "k_max_radial": 9,
    "k_max_angular": 9,
    "l_max": 4,
    "l_max_4body": 2,
    "l_max_5body": 1,
    "NN_radial": 100,
    "NN_angular": 30,
    "hidden_dims": [30],
}


def build_nep_para(elements, para_overrides=None):
    if not elements:
        raise ValueError("`elements` must be provided explicitly to keep type id order stable.")

    para = dict(DEFAULT_MODEL_PARA)
    if para_overrides:
        para.update(para_overrides)
    para["elements"] = list(elements)
    para["n_types"] = len(para["elements"])
    return para


def train_nep(
    train_xyz,
    save_path,
    elements,
    optimizer_name="adam",
    batch_size=5,
    epochs=500,
    generations=1000,
    lr=1e-3,
    snes_population=50,
    snes_sigma=0.1,
    snes_patience=100,
    device=None,
    para_overrides=None,
):
    para = build_nep_para(elements=elements, para_overrides=para_overrides)
    frames = read_xyz(str(train_xyz))
    if len(frames) == 0:
        raise ValueError(f"No frames found in {train_xyz}")

    opt_name = optimizer_name.lower()
    train_dataset = StructureDataset(frames=frames, para=para)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(opt_name == "adam"),
        collate_fn=collate_fn,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training data: {train_xyz}")
    print(f"Element order (type id): {para['elements']}")
    print(f"Optimizer: {opt_name}")
    print(f"Device: {device}")

    model = NEP(para)

    if opt_name == "adam":
        trainer = Optimizer(
            model=model,
            training_set=train_loader,
            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            save_path=str(save_path),
            device=device,
            use_wandb=False,
            compute_q_scaler_once=True,
        )
        trainer.fit(epochs=epochs)
    elif opt_name == "snes":
        trainer = SNES(
            model=model,
            training_set=train_loader,
            save_path=str(save_path),
            device=device,
            population_size=snes_population,
            sigma_init=snes_sigma,
            patience=snes_patience,
            compute_q_scaler_once=True,
        )
        trainer.fit(generations=generations)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'adam' or 'snes'.")

    return model
