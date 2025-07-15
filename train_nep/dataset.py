import torch
from ase.neighborlist import neighbor_list
from ase.data import atomic_numbers
from torch.utils.data import Dataset

def find_neighbor(atoms, cutoff):
    i, j = neighbor_list('ij', atoms, cutoff)
    n_atoms = len(atoms)
    neighbors = [[] for _ in range(n_atoms)]
    for idx in range(len(i)):
        neighbors[i[idx]].append(j[idx])
    return neighbors

def pad_neighbors(neighbors, max_nbs=None, pad_value=-1):
    if max_nbs is None:
        max_nbs = max(len(nbs) for nbs in neighbors)
    padded = []
    for nbs in neighbors:
        arr = list(nbs) + [pad_value] * (max_nbs - len(nbs))
        padded.append(arr)
    return torch.tensor(padded, dtype=torch.long)

def collate_fn(batch):
    atom_offset = 0
    types_list = []
    positions_list = []
    radial_neighbors_list = []
    angular_neighbors_list = []
    
    n_atoms_per_structure = []
    
    energy_list = []
    is_energy = []
    forces_list = []
    virial_list = []
    is_virial = []

    for item in batch:
        types_list.append(item["types"])
        positions_list.append(item["positions"])
        n_atoms_per_structure.append(item["n_atoms"])

        nbs_rad = item["radial_neighbors"].clone()
        valid_rad = nbs_rad != -1
        nbs_rad[valid_rad] += atom_offset
        radial_neighbors_list.append(nbs_rad)

        nbs_ang = item["angular_neighbors"].clone()
        valid_ang = nbs_ang != -1
        nbs_ang[valid_ang] += atom_offset
        angular_neighbors_list.append(nbs_ang)

        e = item.get("energy", None)
        energy_list.append(e)
        is_energy.append(e is not None)

        f = item.get("forces", None)
        forces_list.append(f)

        v = item.get("virial", None)
        virial_list.append(v)
        is_virial.append(v is not None)

        atom_offset += item["n_atoms"]

    types = torch.cat(types_list, dim=0)
    positions = torch.cat(positions_list, dim=0)
    radial_neighbors = torch.cat(radial_neighbors_list, dim=0)
    angular_neighbors = torch.cat(angular_neighbors_list, dim=0)

    result = {
        "types": types,                                 # [N_atoms_total]
        "positions": positions,                         # [N_atoms_total, 3]
        "n_atoms_per_structure": torch.tensor(n_atoms_per_structure, dtype=torch.long), 
        "batch_size": len(batch),                       # int
        "radial_neighbors": radial_neighbors,           # [N_atoms_total, NN_radial]
        "angular_neighbors": angular_neighbors,         # [N_atoms_total, NN_angular]
    }
    
    if any(is_energy):
        result["energy"] = torch.stack([e if e is not None else torch.tensor(0.0, dtype=torch.float32) for e in energy_list])
        result["is_energy"] = torch.tensor(is_energy, dtype=torch.bool)
    
    result["forces"] = torch.cat(forces_list, dim=0)
    
    if any(is_virial):
        result["virial"] = torch.stack([v if v is not None else torch.zeros(9, dtype=torch.float32) for v in virial_list])
        result["is_virial"] = torch.tensor(is_virial, dtype=torch.bool)

    return result

class StructureDataset(Dataset):
    def __init__(self, frames, para):
        self.frames = frames
        self.para = para
        self.elements = para["elements"]
        self.cutoff_radial = para["rcut_radial"]
        self.cutoff_angular = para["rcut_angular"]
        self.NN_radial = para["NN_radial"]
        self.NN_angular = para["NN_angular"]
        element_atomic_numbers = [atomic_numbers[element] for element in self.elements]  
        self.z2id = {z: idx for idx, z in enumerate(element_atomic_numbers)}     
        self.id2z = {idx: z for idx, z in enumerate(element_atomic_numbers)}    
        self.data = [self.process(atoms) for atoms in self.frames]
    
    def process(self, atoms):
        n_atoms = len(atoms)
        atomic_number_list = atoms.get_atomic_numbers()  
        types = [self.z2id[int(z)] for z in atomic_number_list]
        types = torch.tensor(types, dtype=torch.long)
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)

        neighbors_rad = find_neighbor(atoms, self.cutoff_radial)
        neighbors_rad_pad = pad_neighbors(neighbors_rad, max_nbs=self.NN_radial)
       
        neighbors_ang = find_neighbor(atoms, self.cutoff_angular)
        neighbors_ang_pad = pad_neighbors(neighbors_ang, max_nbs=self.NN_angular)
       
        result = {
            "n_atoms": n_atoms,                       # int     
            "types": types,                           # [N_atoms]      
            "positions": positions,                   # [N_atoms, 3]
            "radial_neighbors": neighbors_rad_pad,    # [N_atoms, NN_radial]
            "angular_neighbors": neighbors_ang_pad,   # [N_atoms, NN_angular]
        }
        
        if 'energy' in atoms.info and atoms.info['energy'] is not None:
            result["energy"] = torch.tensor(atoms.info['energy'], dtype=torch.float32)
        
        # check forces
        if 'forces' not in atoms.info or atoms.info['forces'] is None:
            raise ValueError(f"Forces data is missing for structure with {n_atoms} atoms")
        
        forces = atoms.info['forces']
        if not hasattr(forces, '__len__') or len(forces) != n_atoms:
            raise ValueError(f"Forces shape mismatch: expected {n_atoms} atoms, got forces shape {getattr(forces, 'shape', len(forces) if hasattr(forces, '__len__') else 'unknown')}")
        
        forces = torch.tensor(forces, dtype=torch.float32)  # [N_atoms, 3]
        if forces.shape != (n_atoms, 3):
            raise ValueError(f"Forces shape incorrect: expected ({n_atoms}, 3), got {forces.shape}")
        
        result["forces"] = forces
        
        if 'stress' in atoms.info and atoms.info['stress'] is not None:
            stress = atoms.info['stress']
            if len(stress) == 6:
                virial = -stress[[0, 5, 4, 5, 1, 3, 4, 3, 2]] * atoms.get_volume()
            else:
                virial = -stress.reshape(-1) * atoms.get_volume()
            result["virial"] = torch.tensor(virial, dtype=torch.float32)  # [9]
        
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]