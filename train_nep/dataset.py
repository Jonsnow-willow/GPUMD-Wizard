import torch
from ase.neighborlist import neighbor_list
from ase.data import atomic_numbers
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
    atom_offset = 0
    types_list = []
    neighbors_radial_list = []
    distances_radial_list = []
    
    neighbors_angular_list = []
    distances_angular_list = []
    
    n_atoms_per_structure = []

    for item in batch:
        types_list.append(item["types"])
        n_atoms_per_structure.append(item["n_atoms"])

        nbs = item["neighbors_radial"].clone()
        valid = nbs != -1
        nbs[valid] += atom_offset
        neighbors_radial_list.append(nbs)
        distances_radial_list.append(item["distances_radial"])
        
        nbs_ang = item["neighbors_angular"].clone()
        valid_ang = nbs_ang != -1
        nbs_ang[valid_ang] += atom_offset
        neighbors_angular_list.append(nbs_ang)
        distances_angular_list.append(item["distances_angular"])
        
        atom_offset += item["n_atoms"]

    types = torch.cat(types_list, dim=0)
    neighbors_radial = torch.cat(neighbors_radial_list, dim=0)
    distances_radial = torch.cat(distances_radial_list, dim=0)
    neighbors_angular = torch.cat(neighbors_angular_list, dim=0)
    distances_angular = torch.cat(distances_angular_list, dim=0)

    triplet_indices = []
    r_ij_list = []
    r_ik_list = []
    cos_theta_list = []
    type_i_list = []
    type_j_list = []
    type_k_list = []
    
    n_atoms_total = types.shape[0]
    
    for atom_i in range(n_atoms_total):
        neighbors_i = neighbors_angular[atom_i]
        distances_i = distances_angular[atom_i]  # [NN_angular, 3]
        
        valid_mask = neighbors_i != -1
        valid_neighbors = neighbors_i[valid_mask]
        valid_distances = distances_i[valid_mask]  # [n_valid, 3]
        
        if len(valid_neighbors) < 2:
            continue
            
        for j_idx in range(len(valid_neighbors)):
            for k_idx in range(j_idx + 1, len(valid_neighbors)):
                j = valid_neighbors[j_idx].item()
                k = valid_neighbors[k_idx].item()
                
                rij_vec = valid_distances[j_idx]  # [3]
                rik_vec = valid_distances[k_idx]  # [3]
                
                rij = torch.norm(rij_vec)
                rik = torch.norm(rik_vec)
                cos_theta = torch.dot(rij_vec, rik_vec) / (rij * rik + 1e-8)
                
                triplet_indices.append([atom_i, j, k])
                r_ij_list.append(rij)
                r_ik_list.append(rik)
                cos_theta_list.append(cos_theta)
                type_i_list.append(types[atom_i])
                type_j_list.append(types[j])
                type_k_list.append(types[k])

    triplet_index = torch.tensor(triplet_indices, dtype=torch.long) if triplet_indices else torch.zeros((0, 3), dtype=torch.long)
    r_ij = torch.stack(r_ij_list) if r_ij_list else torch.zeros(0, dtype=torch.float32)
    r_ik = torch.stack(r_ik_list) if r_ik_list else torch.zeros(0, dtype=torch.float32)
    cos_theta = torch.stack(cos_theta_list) if cos_theta_list else torch.zeros(0, dtype=torch.float32)
    type_i = torch.tensor(type_i_list, dtype=torch.long) if type_i_list else torch.zeros(0, dtype=torch.long)
    type_j = torch.tensor(type_j_list, dtype=torch.long) if type_j_list else torch.zeros(0, dtype=torch.long)
    type_k = torch.tensor(type_k_list, dtype=torch.long) if type_k_list else torch.zeros(0, dtype=torch.long)

    return {
        "types": types,                                 # [N_atoms_total]
        "n_atoms_per_structure": torch.tensor(n_atoms_per_structure, dtype=torch.long), 
        "batch_size": len(batch),                       # int

        "radial_neighbors": neighbors_radial,           # [N_atoms_total, NN_radial]
        "radial_distances": distances_radial,           # [N_atoms_total, NN_radial]
        
        "triplet_index": triplet_index,                 # [N_triplets, 3]
        "r_ij": r_ij,                                   # [N_triplets]
        "r_ik": r_ik,                                   # [N_triplets]
        "cos_theta": cos_theta,                         # [N_triplets]
        "type_i": type_i,                               # [N_triplets]
        "type_j": type_j,                               # [N_triplets]
        "type_k": type_k,                               # [N_triplets]
    }

class StructureDataset(Dataset):
    def __init__(self, filepath, types, cutoff_radial, cutoff_angular, NN_radial, NN_angular):
        self.structures = read_xyz(filepath)
        self.cutoff_radial = cutoff_radial
        self.cutoff_angular = cutoff_angular
        self.NN_radial = NN_radial
        self.NN_angular = NN_angular

        element_names = types.split()   
        element_z = [atomic_numbers[sym] for sym in element_names]  
        self.z2id = {z: idx for idx, z in enumerate(element_z)}     
        self.id2z = {idx: z for idx, z in enumerate(element_z)}    
        self.data = [self.process(atoms) for atoms in self.structures]
    
    def process(self, atoms):
        n_atoms = len(atoms)
        raw_types = atoms.get_atomic_numbers()  
        types = [self.z2id[int(z)] for z in raw_types]
        types = torch.tensor(types, dtype=torch.long)

        neighbors_rad, distances_rad = find_neighbor_radial(atoms, self.cutoff_radial)
        neighbors_rad_pad = pad_neighbors(neighbors_rad, max_nbs=self.NN_radial)
        distances_rad_pad = pad_distances(distances_rad, max_nbs=self.NN_radial)

        neighbors_ang, distances_ang = find_neighbor_angular(atoms, self.cutoff_angular)
        neighbors_ang_pad = pad_neighbors(neighbors_ang, max_nbs=self.NN_angular)
        distances_ang_pad = pad_distances(distances_ang, max_nbs=self.NN_angular, pad_value=[0, 0, 0])

        return {
            "n_atoms": n_atoms,   
            "types": types,  
            "neighbors_radial": neighbors_rad_pad,
            "distances_radial": distances_rad_pad,
            "neighbors_angular": neighbors_ang_pad,
            "distances_angular": distances_ang_pad,  # [N_atoms, NN_angular, 3]
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]