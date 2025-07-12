import torch
import torch.nn as nn

def chebyshev_basis(r, r_c, k_max):
    fc = torch.where(
        r < r_c,
        0.5 * torch.cos(torch.pi * r / r_c) + 0.5,
        torch.zeros_like(r)
    )
    x = 2.0 * (r / r_c - 1.0) ** 2 - 1.0
    half_fc = 0.5 * fc
    fn = [torch.ones_like(x)]
    if k_max > 1:
        fn.append(x)
        for _ in range(2, k_max):
            temp = 2 * x * fn[-1] - fn[-2]   
            fn.append(temp)
    fn = torch.stack(fn, dim=-1)
    fn = (fn + 1.0) * half_fc.unsqueeze(-1)
    return fn

def legendre_basis(cos_theta, l_max):
    pl = [torch.ones_like(cos_theta), cos_theta]
    for l in range(2, l_max):
        temp = ((2 * l - 1) * cos_theta * pl[-1] - (l - 1) * pl[-2]) / l
        pl.append(temp)
    pl = torch.stack(pl[:l_max], dim=-1)
    return pl

class RadialDescriptor(nn.Module):
    def __init__(self, n_types, n_desc, k_max, r_c):
        super().__init__()
        self.n_types = n_types
        self.n_desc = n_desc
        self.k_max = k_max
        self.r_c = r_c
        self.c_table = nn.Parameter(
            torch.randn(n_types, n_types, n_desc, k_max)
        )

    def get_attention(self, type_i, type_j):
        return self.c_table[type_i, type_j]  
    
    def forward(self, types, positions, radial_neighbors):
        n_atoms, nn_radial = radial_neighbors.shape
        valid_mask = radial_neighbors != -1  # [N_atoms, NN_radial]
        
        if not valid_mask.any():
            return torch.zeros(n_atoms, self.n_desc, device=types.device)

        atom_indices = torch.arange(n_atoms, device=types.device).unsqueeze(1).expand(-1, nn_radial)  # [N_atoms, NN_radial]
        
        valid_atom_indices = atom_indices[valid_mask]  # [N_valid_edges]
        valid_neighbor_indices = radial_neighbors[valid_mask]  # [N_valid_edges]
        
        positions_i = positions[valid_atom_indices]  # [N_valid_edges, 3]
        positions_j = positions[valid_neighbor_indices]  # [N_valid_edges, 3]
        type_i = types[valid_atom_indices]  # [N_valid_edges]
        type_j = types[valid_neighbor_indices]  # [N_valid_edges]
        
        radial_distances = torch.norm(positions_j - positions_i, dim=-1)  # [N_valid_edges]
        
        f = chebyshev_basis(radial_distances, self.r_c, self.k_max)  # [N_valid_edges, k_max]
        c = self.get_attention(type_i, type_j)                      # [N_valid_edges, n_desc, k_max]
        edge_descriptors = torch.sum(c * f.unsqueeze(1), dim=-1)    # [N_valid_edges, n_desc]
        
        g = torch.zeros(n_atoms, self.n_desc, device=types.device)
        g.index_add_(0, valid_atom_indices, edge_descriptors)
        
        return g  # [N_atoms, n_desc]

class AngularDescriptor(nn.Module):
    def __init__(self, n_types, n_desc, k_max, r_c, l_max):
        super().__init__()
        self.n_types = n_types
        self.n_desc = n_desc
        self.k_max = k_max
        self.r_c = r_c
        self.l_max = l_max
        
        self.c_table = nn.Parameter(
            torch.randn(n_types, n_types, n_desc, k_max)
        )

    def get_attention(self, type_i, type_j):
        return self.c_table[type_i, type_j]
    
    def compute_radial_for_triplets(self, r, type_i, type_j):
        f = chebyshev_basis(r, self.r_c, self.k_max)     
        c = self.get_attention(type_i, type_j)           
        g = torch.sum(c * f.unsqueeze(1), dim=-1)        
        return g

    def build_triplets_from_neighbors(self, angular_neighbors, types):
        n_atoms, _ = angular_neighbors.shape
        valid_mask = angular_neighbors != -1  # [N_atoms, NN_angular]
        
        triplet_indices = []
        
        for atom_i in range(n_atoms):
            neighbors_i = angular_neighbors[atom_i]
            valid_neighbors_i = neighbors_i[valid_mask[atom_i]]  # 只取有效邻居
            
            if len(valid_neighbors_i) < 2:
                continue
                
            for j_idx in range(len(valid_neighbors_i)):
                for k_idx in range(j_idx + 1, len(valid_neighbors_i)):
                    j = valid_neighbors_i[j_idx].item()
                    k = valid_neighbors_i[k_idx].item()
                    triplet_indices.append([atom_i, j, k])
        
        return torch.tensor(triplet_indices, dtype=torch.long, device=types.device) if triplet_indices else torch.zeros((0, 3), dtype=torch.long, device=types.device)

    def forward(self, types, positions, angular_neighbors):
        triplet_index = self.build_triplets_from_neighbors(angular_neighbors, types)
        
        if triplet_index.shape[0] == 0:
            n_atoms = positions.shape[0]
            return torch.zeros(n_atoms, self.n_desc, self.l_max, device=positions.device)
        
        i_idx = triplet_index[:, 0]  # [N_triplets]
        j_idx = triplet_index[:, 1]  # [N_triplets]
        k_idx = triplet_index[:, 2]  # [N_triplets]
        
        type_i = types[i_idx]  # [N_triplets]
        type_j = types[j_idx]  # [N_triplets]
        type_k = types[k_idx]  # [N_triplets]
        
        pos_i = positions[i_idx]  # [N_triplets, 3]
        pos_j = positions[j_idx]  # [N_triplets, 3]
        pos_k = positions[k_idx]  # [N_triplets, 3]
        
        rij_vec = pos_j - pos_i  # [N_triplets, 3]
        rik_vec = pos_k - pos_i  # [N_triplets, 3]
        
        r_ij = torch.norm(rij_vec, dim=-1)  # [N_triplets]
        r_ik = torch.norm(rik_vec, dim=-1)  # [N_triplets]
        
        cos_theta = torch.sum(rij_vec * rik_vec, dim=-1) / (r_ij * r_ik + 1e-8)  # [N_triplets]
        
        g_ij = self.compute_radial_for_triplets(r_ij, type_i, type_j)  # [N_triplets, n_desc]
        g_ik = self.compute_radial_for_triplets(r_ik, type_i, type_k)  # [N_triplets, n_desc]

        P_l = legendre_basis(cos_theta, self.l_max)                    # [N_triplets, l_max]
        ang = g_ij * g_ik                                              # [N_triplets, n_desc]
        ang = ang.unsqueeze(-1) * P_l.unsqueeze(1)                     # [N_triplets, n_desc, l_max]

        n_atoms = positions.shape[0]
        q = torch.zeros(n_atoms, self.n_desc, self.l_max, device=ang.device)
        q.index_add_(0, i_idx, ang)  

        return q 

class Descriptor(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.radial = RadialDescriptor(
            n_types=para["n_types"],
            n_desc=para["n_desc_radial"],
            k_max=para["k_max_radial"],
            r_c=para["rcut_radial"]  
        )
        self.angular = AngularDescriptor(
            n_types=para["n_types"],
            n_desc=para["n_desc_angular"],
            k_max=para["k_max_angular"],
            r_c=para["rcut_angular"],
            l_max=para["l_max"]
        )

    def forward(self, batch):
        g_radial = self.radial(
            batch["types"],              # [N_atoms]
            batch["positions"],          # [N_atoms, 3]
            batch["radial_neighbors"],   # [N_atoms, NN_radial] padding tensor
        )  # [N_atoms, n_desc_radial]
        
        g_angular = self.angular(
            batch["types"],               # [N_atoms]
            batch["positions"],           # [N_atoms, 3]
            batch["angular_neighbors"]    # [N_atoms, NN_angular] padding tensor
        )  # [n_atoms, n_desc_angular, l_max]

        return {
            "g_radial": g_radial,
            "g_angular": g_angular
        }