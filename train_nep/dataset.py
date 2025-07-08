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
    types = [item["types"] for item in batch]
    neighbors = [item["neighbors"] for item in batch]
    distances = [item["distances"] for item in batch]
    
    n_atoms = [len(pos) for pos in positions]
    
    positions_batch = torch.cat(positions, dim=0)
    types_batch = torch.cat(types, dim=0)
    
    neighbors_batch = []
    distances_batch = []
    atom_offset = 0
    
    for i, (nbs, dists) in enumerate(zip(neighbors, distances)):
        valid_mask = nbs != -1
        nbs_updated = nbs.clone()
        nbs_updated[valid_mask] += atom_offset
        neighbors_batch.append(nbs_updated)
        distances_batch.append(dists)
        atom_offset += n_atoms[i]
    
    neighbors_batch = torch.cat(neighbors_batch, dim=0)
    distances_batch = torch.cat(distances_batch, dim=0)
    
    return {
        "positions": positions_batch,
        "types": types_batch,
        "neighbors": neighbors_batch,
        "distances": distances_batch,
        "n_atoms": torch.tensor(n_atoms),
        "batch_size": len(batch)
    }

class StructureDataset(Dataset):
    def __init__(self, filepath, cutoff):
        self.structures = read_xyz(filepath)
        self.cutoff = cutoff
        self.data = [self.process(atoms) for atoms in self.structures]
    
    def process(self, atoms):
        coords = atoms.get_positions()
        types = atoms.get_atomic_numbers()
        neighbors, distances = find_neighbor(atoms, self.cutoff)
        neighbors_pad = pad_neighbors(neighbors)
        distances_pad = pad_distances(distances)
        return {
            "positions": torch.tensor(coords, dtype=torch.float32),
            "types": torch.tensor(types, dtype=torch.long),
            "neighbors": neighbors_pad,
            "distances": distances_pad,
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]