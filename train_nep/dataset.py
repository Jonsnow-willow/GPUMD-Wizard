from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset
from wizard.io import read_xyz
import torch

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

def pad_distances(distances, max_nbs=None, pad_value=0.0):
    """Pad distance lists to the same length."""
    if max_nbs is None:
        max_nbs = max(len(dists) for dists in distances)
    padded = []
    for dists in distances:
        arr = list(dists) + [pad_value] * (max_nbs - len(dists))
        padded.append(arr)
    return torch.tensor(padded, dtype=torch.float32)

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
            "positions": torch.tensor(coords, dtype=torch.float32),      # (n_atoms, 3)
            "types": torch.tensor(types, dtype=torch.long),              # (n_atoms,)
            "neighbors": neighbors_pad,                                  # (n_atoms, max_nbs)
            "distances": distances_pad,                                  # (n_atoms, max_nbs)
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

