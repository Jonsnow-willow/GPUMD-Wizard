import torch
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset
from wizard.io import read_xyz

def find_neighbor_radial(atoms, cutoff):
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    n_atoms = len(atoms)
    neighbors = [[] for _ in range(n_atoms)]
    distances = [[] for _ in range(n_atoms)]
    for idx in range(len(i)):
        neighbors[i[idx]].append(j[idx])
        distances[i[idx]].append(d[idx])
    return neighbors, distances

def find_neighbor_angular(atoms, cutoff):
    i, j, D = neighbor_list('ijD', atoms, cutoff)
    n_atoms = len(atoms)
    neighbors = [[] for _ in range(n_atoms)]
    distances = [[] for _ in range(n_atoms)]
    for idx in range(len(i)):
        neighbors[i[idx]].append(j[idx])
        distances[i[idx]].append(D[idx]) # [NN_angular, 3]
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
    
    triplet_indices = []
    r_ij_list = []
    r_ik_list = []
    cos_theta_list = []
    type_i_list = []
    type_j_list = []
    type_k_list = []
    
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
        dists_ang = distances_angular[i]  # [N_atoms, NN_angular, 3]
        valid_mask_ang = nbs_ang != -1
        nbs_ang_updated = nbs_ang.clone()
        nbs_ang_updated[valid_mask_ang] += atom_offset
        neighbors_angular_batch.append(nbs_ang_updated)
        distances_angular_batch.append(dists_ang)

        types_i = types[i]
        
        for atom_idx in range(n_atoms[i]):
            global_atom_idx = atom_offset + atom_idx
            neighbors = nbs_ang[atom_idx]
            
            valid_neighbors = neighbors[neighbors != -1]
            if len(valid_neighbors) < 2:
                continue
                
            for j_idx, j in enumerate(valid_neighbors):
                for k_idx, k in enumerate(valid_neighbors):
                    if j_idx >= k_idx:  
                        continue
                    
                    rij_vec = dists_ang[atom_idx, j_idx]  
                    rik_vec = dists_ang[atom_idx, k_idx]  
                    
                    rij = torch.norm(rij_vec)
                    rik = torch.norm(rik_vec)
                    cos_theta = torch.dot(rij_vec, rik_vec) / (rij * rik + 1e-8)
                    
                    triplet_indices.append([global_atom_idx, atom_offset + j, atom_offset + k])
                    r_ij_list.append(rij)
                    r_ik_list.append(rik)
                    cos_theta_list.append(cos_theta)
                    type_i_list.append(types_i[atom_idx])
                    type_j_list.append(types_i[j])
                    type_k_list.append(types_i[k])

        atom_offset += n_atoms[i]

    neighbors_radial_batch = torch.cat(neighbors_radial_batch, dim=0)
    distances_radial_batch = torch.cat(distances_radial_batch, dim=0)
    neighbors_angular_batch = torch.cat(neighbors_angular_batch, dim=0)
    distances_angular_batch = torch.cat(distances_angular_batch, dim=0)

    triplet_index = torch.tensor(triplet_indices, dtype=torch.long) if triplet_indices else torch.zeros((0, 3), dtype=torch.long)
    r_ij = torch.stack(r_ij_list) if r_ij_list else torch.zeros(0, dtype=torch.float32)
    r_ik = torch.stack(r_ik_list) if r_ik_list else torch.zeros(0, dtype=torch.float32)
    cos_theta = torch.stack(cos_theta_list) if cos_theta_list else torch.zeros(0, dtype=torch.float32)
    type_i = torch.tensor(type_i_list, dtype=torch.long) if type_i_list else torch.zeros(0, dtype=torch.long)
    type_j = torch.tensor(type_j_list, dtype=torch.long) if type_j_list else torch.zeros(0, dtype=torch.long)
    type_k = torch.tensor(type_k_list, dtype=torch.long) if type_k_list else torch.zeros(0, dtype=torch.long)

    return {
        "positions": positions_batch,                   # [N_atoms_total, 3]
        "types": types_batch,                           # [N_atoms_total]
        "neighbors_radial": neighbors_radial_batch,     # [N_atoms_total, NN_radial]
        "distances_radial": distances_radial_batch,     # [N_atoms_total, NN_radial]
        "neighbors_angular": neighbors_angular_batch,   # [N_atoms_total, NN_angular]
        "distances_angular": distances_angular_batch,   # [N_atoms_total, NN_angular, 3]
        "n_atoms": torch.tensor(n_atoms),
        "batch_size": len(batch),
        "triplet_index": triplet_index,                 # [N_triplets, 3]
        "r_ij": r_ij,                                   # [N_triplets]
        "r_ik": r_ik,                                   # [N_triplets]
        "cos_theta": cos_theta,                         # [N_triplets]
        "type_i": type_i,                               # [N_triplets]
        "type_j": type_j,                               # [N_triplets]
        "type_k": type_k,                               # [N_triplets]
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

        neighbors_rad, distances_rad = find_neighbor_radial(atoms, self.cutoff_radial)
        neighbors_rad_pad = pad_neighbors(neighbors_rad, max_nbs=self.NN_radial)
        distances_rad_pad = pad_distances(distances_rad, max_nbs=self.NN_radial)

        neighbors_ang, distances_ang = find_neighbor_angular(atoms, self.cutoff_angular)
        neighbors_ang_pad = pad_neighbors(neighbors_ang, max_nbs=self.NN_angular)
        distances_ang_pad = pad_distances(distances_ang, max_nbs=self.NN_angular, pad_value=[0, 0, 0])

        return {
            "positions": torch.tensor(coords, dtype=torch.float32),
            "types": torch.tensor(types, dtype=torch.long),
            "neighbors_radial": neighbors_rad_pad,
            "distances_radial": distances_rad_pad,
            "neighbors_angular": neighbors_ang_pad,
            "distances_angular": distances_ang_pad,  # [N_atoms, NN_angular, 3]
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]