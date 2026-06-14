from __future__ import annotations

import hashlib
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.data import atomic_numbers
from torch.utils.data import Dataset

from wizard.torchNEP.datasets.neighbor import build_neighbor_list


class LazyExtendedXYZDataset(Dataset):
    def __init__(
        self,
        xyz_path: str | Path,
        para: dict[str, Any],
        index_path: str | Path | None = None,
        cache_index: bool = True,
        frame_stride: int = 1,
        max_frames: int | None = None,
        require_forces: bool = True,
    ):
        self.xyz_path = Path(xyz_path)
        self.para = para
        self.require_forces = require_forces
        self.elements = para["elements"]
        self.cutoff_radial = para["rcut_radial"]
        self.cutoff_angular = para["rcut_angular"]
        self.NN_radial = para["NN_radial"]
        self.NN_angular = para["NN_angular"]
        self.z2id = {
            atomic_numbers[element]: idx
            for idx, element in enumerate(self.elements)
        }
        self.index_path = Path(index_path) if index_path is not None else None
        self.offsets, self.n_atoms = load_or_build_index(
            self.xyz_path,
            index_path=self.index_path,
            cache_index=cache_index,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        if len(self.offsets) == 0:
            raise ValueError(f"No frames found in {self.xyz_path}")
        self._file = None

    def __len__(self):
        return int(len(self.offsets))

    def __getitem__(self, idx):
        frame = self._read_frame(int(self.offsets[idx]))
        return self._process_frame(frame)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def _get_file(self):
        if self._file is None:
            self._file = self.xyz_path.open("rb")
        return self._file

    def _read_frame(self, offset: int) -> dict[str, Any]:
        f = self._get_file()
        f.seek(offset)
        natoms = int(f.readline().strip())
        comment = f.readline().decode("utf-8").strip()
        metadata = parse_comment(comment)
        properties = parse_properties(metadata)
        symbols = []
        positions = np.empty((natoms, 3), dtype=np.float32)
        forces = np.empty((natoms, 3), dtype=np.float32) if has_force_property(properties) else None

        for atom_idx in range(natoms):
            tokens = f.readline().decode("utf-8").split()
            symbols.append(read_symbol(tokens, properties))
            positions[atom_idx] = read_vector(tokens, properties, "pos")
            if forces is not None:
                forces[atom_idx] = read_vector(tokens, properties, force_property_name(properties))

        return {
            "symbols": symbols,
            "positions": positions,
            "cell": read_cell(metadata),
            "pbc": read_pbc(metadata),
            "energy": read_float(metadata, "energy"),
            "stress": read_stress(metadata),
            "forces": forces,
        }

    def _process_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        n_atoms = len(frame["symbols"])
        types = []
        missing_symbols = []
        for symbol in frame["symbols"]:
            z = atomic_numbers[symbol]
            type_id = self.z2id.get(z)
            if type_id is None:
                missing_symbols.append(symbol)
            else:
                types.append(type_id)
        if missing_symbols:
            missing = sorted(set(missing_symbols))
            raise ValueError(
                "Dataset contains elements not listed in para['elements']. "
                f"Missing elements: {missing}. Provided elements: {self.elements}."
            )

        (
            neighbors_rad,
            offsets_rad,
            neighbors_ang,
            offsets_ang,
        ) = build_neighbor_list(
            positions=frame["positions"],
            cell=frame["cell"],
            pbc=frame["pbc"],
            cutoff_radial=self.cutoff_radial,
            cutoff_angular=self.cutoff_angular,
            max_neighbors_radial=self.NN_radial,
            max_neighbors_angular=self.NN_angular,
        )

        result = {
            "n_atoms": n_atoms,
            "types": torch.tensor(types, dtype=torch.long),
            "positions": torch.from_numpy(frame["positions"]).float(),
            "radial_neighbors": torch.from_numpy(neighbors_rad).long(),
            "angular_neighbors": torch.from_numpy(neighbors_ang).long(),
            "radial_offsets": torch.from_numpy(offsets_rad).float(),
            "angular_offsets": torch.from_numpy(offsets_ang).float(),
        }

        if frame["energy"] is not None:
            result["energy"] = torch.tensor(frame["energy"], dtype=torch.float32)

        if frame["forces"] is None:
            if self.require_forces:
                raise ValueError(f"Forces data is missing for structure with {n_atoms} atoms")
        else:
            result["forces"] = torch.from_numpy(frame["forces"]).float()

        if frame["stress"] is not None:
            volume = abs(float(np.linalg.det(frame["cell"])))
            result["virial"] = torch.tensor((-frame["stress"].reshape(-1) * volume), dtype=torch.float32)

        return result


def load_or_build_index(
    xyz_path: Path,
    index_path: Path | None,
    cache_index: bool,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")
    metadata = index_metadata(xyz_path, frame_stride=frame_stride, max_frames=max_frames)
    if cache_index and index_path is not None and index_path.exists():
        cached = np.load(index_path, allow_pickle=False)
        if cached_metadata_matches(cached, metadata):
            return cached["offsets"], cached["n_atoms"]

    offsets, n_atoms = scan_xyz_offsets(xyz_path, frame_stride=frame_stride, max_frames=max_frames)
    if cache_index and index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(index_path, offsets=offsets, n_atoms=n_atoms, **metadata)
    return offsets, n_atoms


def scan_xyz_offsets(
    xyz_path: Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = []
    n_atoms = []
    seen_frames = 0
    with xyz_path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            natoms = int(line.strip())
            f.readline()
            selected = seen_frames % frame_stride == 0
            if selected:
                offsets.append(offset)
                n_atoms.append(natoms)
                if max_frames is not None and len(offsets) >= max_frames:
                    break
            for _ in range(natoms):
                f.readline()
            seen_frames += 1
    return np.asarray(offsets, dtype=np.int64), np.asarray(n_atoms, dtype=np.int32)


def default_index_path(run_dir: str | Path, index_dir: str | Path, xyz_path: str | Path, split: str) -> Path:
    xyz_path = Path(xyz_path).resolve()
    digest = hashlib.sha1(str(xyz_path).encode("utf-8")).hexdigest()[:16]
    index_dir = Path(index_dir)
    if not index_dir.is_absolute():
        index_dir = Path(run_dir) / index_dir
    return index_dir / f"{split}-{xyz_path.name}-{digest}.npz"


def index_metadata(xyz_path: Path, frame_stride: int, max_frames: int | None) -> dict[str, Any]:
    stat = xyz_path.stat()
    return {
        "file_size": np.asarray(stat.st_size, dtype=np.int64),
        "file_mtime_ns": np.asarray(stat.st_mtime_ns, dtype=np.int64),
        "frame_stride": np.asarray(frame_stride, dtype=np.int64),
        "max_frames": np.asarray(-1 if max_frames is None else max_frames, dtype=np.int64),
    }


def cached_metadata_matches(cached, expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key not in cached or int(cached[key]) != int(value):
            return False
    return True


def parse_comment(comment: str) -> dict[str, str]:
    metadata = {}
    for token in shlex.split(comment):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        metadata[key.lower()] = value
    return metadata


def parse_properties(metadata: dict[str, str]) -> dict[str, slice]:
    properties = metadata.get("properties")
    if properties is None:
        raise ValueError("Extended XYZ comment is missing Properties=...")
    tokens = properties.split(":")
    parsed = {}
    start = 0
    for idx in range(0, len(tokens), 3):
        name = tokens[idx].lower()
        count = int(tokens[idx + 2])
        parsed[name] = slice(start, start + count)
        start += count
    return parsed


def read_symbol(tokens: list[str], properties: dict[str, slice]) -> str:
    value = tokens[properties["species"]][0]
    return value.lower().capitalize()


def read_vector(tokens: list[str], properties: dict[str, slice], key: str) -> tuple[float, float, float]:
    values = tokens[properties[key]]
    return float(values[0]), float(values[1]), float(values[2])


def read_cell(metadata: dict[str, str]) -> np.ndarray:
    lattice = metadata.get("lattice")
    if lattice is None:
        raise ValueError("Extended XYZ comment is missing Lattice=...")
    values = [float(value) for value in lattice.split()]
    if len(values) != 9:
        raise ValueError("Lattice should contain 9 values.")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def read_pbc(metadata: dict[str, str]) -> list[bool]:
    pbc = metadata.get("pbc")
    if pbc is None:
        return [True, True, True]
    return [value.lower().startswith("t") for value in pbc.split()]


def read_float(metadata: dict[str, str], key: str) -> float | None:
    value = metadata.get(key)
    return None if value is None else float(value)


def read_stress(metadata: dict[str, str]) -> np.ndarray | None:
    value = metadata.get("stress")
    if value is None:
        return None
    values = np.asarray([float(token) for token in value.split()], dtype=np.float64)
    if values.size == 9:
        return values.reshape(3, 3)
    if values.size == 6:
        return values[[0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(3, 3)
    raise ValueError("stress should contain 6 or 9 values.")


def has_force_property(properties: dict[str, slice]) -> bool:
    return "forces" in properties or "force" in properties


def force_property_name(properties: dict[str, slice]) -> str:
    if "forces" in properties:
        return "forces"
    return "force"
