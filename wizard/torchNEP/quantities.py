from __future__ import annotations

import torch


VIRIAL_COMPONENT_INDICES = (0, 4, 8, 1, 2, 5)


def independent_virial_components(virial: torch.Tensor) -> torch.Tensor:
    """Return xx, yy, zz, xy, xz, yz without double-counting shear terms."""
    if virial.ndim >= 2 and virial.shape[-2:] == (3, 3):
        return torch.stack(
            (
                virial[..., 0, 0],
                virial[..., 1, 1],
                virial[..., 2, 2],
                virial[..., 0, 1],
                virial[..., 0, 2],
                virial[..., 1, 2],
            ),
            dim=-1,
        )
    if virial.shape[-1] == 9:
        return virial[..., VIRIAL_COMPONENT_INDICES]
    if virial.shape[-1] == 6:
        return virial
    raise ValueError(f"Expected virial tensors with 6 or 9 components, got shape {tuple(virial.shape)}.")
