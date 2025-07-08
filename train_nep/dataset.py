import torch
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset
from wizard.io import read_xyz

def find_neighbor(atoms, cutoff):
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    n_atoms = len(atoms)
    neighbors = [[] for _ in range(n_atoms)]
    distances = [[] for _ in range(n_atoms)]
    for idx in range(len(i)):
        neighbors[i[idx]].append(j[idx])
        distances[i[idx]].append(d[idx])
    return neighbors, distances

def pad_neighbors(neighbors, max_nbs=None, pad_value=-1):
    if max_nbs is None:
        max_nbs = max(len(nbs) for nbs in neighbors)
    padded = []
    for nbs in neighbors:
        arr = list(nbs) + [pad_value] * (max_nbs - len(nbs))
        padded.append(arr)
    return torch.tensor(padded, dtype=torch.long)

def pad_distances(distances, max_nbs=None, pad_value=-1):
    if max_nbs is None:
        max_nbs = max(len(dists) for dists in distances)
    padded = []
    for dists in distances:
        arr = list(dists) + [pad_value] * (max_nbs - len(dists))
        padded.append(arr)
    return torch.tensor(padded, dtype=torch.float32)

def collate_fn(batch):
    positions = [item["positions"] for item in batch]
    positions_batch = torch.cat(positions, dim=0)

    types = [item["types"] for item in batch]
    types_batch = torch.cat(types, dim=0)

    n_atoms = [len(pos) for pos in positions]
    neighbors_radial = [item["neighbors_radial"] for item in batch]
    distances_radial = [item["distances_radial"] for item in batch]
    neighbors_angular = [item["neighbors_angular"] for item in batch]
    distances_angular = [item["distances_angular"] for item in batch]

    neighbors_radial_batch = []
    distances_radial_batch = []
    neighbors_angular_batch = []
    distances_angular_batch = []
    atom_offset = 0

    for i in range(len(batch)):

        nbs_rad = neighbors_radial[i]
        dists_rad = distances_radial[i]
        valid_mask_rad = nbs_rad != -1
        nbs_rad_updated = nbs_rad.clone()
        nbs_rad_updated[valid_mask_rad] += atom_offset
        neighbors_radial_batch.append(nbs_rad_updated)
        distances_radial_batch.append(dists_rad)

        nbs_ang = neighbors_angular[i]
        dists_ang = distances_angular[i]
        valid_mask_ang = nbs_ang != -1
        nbs_ang_updated = nbs_ang.clone()
        nbs_ang_updated[valid_mask_ang] += atom_offset
        neighbors_angular_batch.append(nbs_ang_updated)
        distances_angular_batch.append(dists_ang)

        atom_offset += n_atoms[i]

    neighbors_radial_batch = torch.cat(neighbors_radial_batch, dim=0)
    distances_radial_batch = torch.cat(distances_radial_batch, dim=0)
    neighbors_angular_batch = torch.cat(neighbors_angular_batch, dim=0)
    distances_angular_batch = torch.cat(distances_angular_batch, dim=0)

    return {
        "positions": positions_batch,
        "types": types_batch,
        "neighbors_radial": neighbors_radial_batch,
        "distances_radial": distances_radial_batch,
        "neighbors_angular": neighbors_angular_batch,
        "distances_angular": distances_angular_batch,
        "n_atoms": torch.tensor(n_atoms),
        "batch_size": len(batch)
    }

class StructureDataset(Dataset):
    def __init__(self, filepath, cutoff_radial, cutoff_angular, NN_radial, NN_angular):
        self.structures = read_xyz(filepath)
        self.cutoff_radial = cutoff_radial
        self.cutoff_angular = cutoff_angular
        self.NN_radial = NN_radial
        self.NN_angular = NN_angular
        self.data = [self.process(atoms) for atoms in self.structures]
    
    def process(self, atoms):
        coords = atoms.get_positions()
        types = atoms.get_atomic_numbers()

        neighbors_rad, distances_rad = find_neighbor(atoms, self.cutoff_radial)
        neighbors_rad_pad = pad_neighbors(neighbors_rad, max_nbs=self.NN_radial)
        distances_rad_pad = pad_distances(distances_rad, max_nbs=self.NN_radial)

        neighbors_ang, distances_ang = find_neighbor(atoms, self.cutoff_angular)
        neighbors_ang_pad = pad_neighbors(neighbors_ang, max_nbs=self.NN_angular)
        distances_ang_pad = pad_distances(distances_ang, max_nbs=self.NN_angular)

        return {
            "positions": torch.tensor(coords, dtype=torch.float32),
            "types": torch.tensor(types, dtype=torch.long),
            "neighbors_radial": neighbors_rad_pad,
            "distances_radial": distances_rad_pad,
            "neighbors_angular": neighbors_ang_pad,
            "distances_angular": distances_ang_pad,
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]