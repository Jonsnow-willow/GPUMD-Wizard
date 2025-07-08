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
    
    def forward(self, r_ij, type_i, type_j):
        f = chebyshev_basis(r_ij, self.r_c, self.k_max)   
        c = self.get_attention(type_i, type_j)            
        # f.unsqueeze(1): [N, 1, k_max], c: [n_desc, k_max]
        g = torch.sum(c * f.unsqueeze(1), dim=-1)         # [N, n_desc]
        return g

class AngularDescriptor(nn.Module):
    def __init__(self, n_types, n_desc, k_max, r_c, l_max):
        super().__init__()
        self.radial_descriptor = RadialDescriptor(n_types, n_desc, k_max, r_c)
        self.l_max = l_max

    def forward(self, n_atoms, triplet_index, r_ij, r_ik, cos_theta, type_i, type_j, type_k):
        g_ij = self.radial_descriptor(r_ij, type_i, type_j)   
        g_ik = self.radial_descriptor(r_ik, type_i, type_k)   

        P_l = legendre_basis(cos_theta, self.l_max)   
        ang = g_ij * g_ik 
        ang = ang.unsqueeze(-1) * P_l.unsqueeze(1)  # [n_triplet, n_desc, l_max+1]

        q = torch.zeros(n_atoms, self.n_desc, self.l_max, device=ang.device)
        i_idx = triplet_index[:, 0]   # [n_triplet]
        q.index_add_(0, i_idx, ang)  

        return q  # [n_atoms, n_desc, l_max+1]

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
            batch["r_ij_radial"],    
            batch["type_i_radial"],      
            batch["type_j_radial"]        
        ) 
        
        n_atoms_total = torch.sum(batch["n_atoms"]).item()
        g_angular = self.angular(
            n_atoms_total,                # int
            batch["triplet_index"],       # [N_triplets, 3]
            batch["r_ij"],                # [N_triplets]
            batch["r_ik"],                # [N_triplets]
            batch["cos_theta"],           # [N_triplets]
            batch["type_i"],              # [N_triplets]
            batch["type_j"],              # [N_triplets]
            batch["type_k"]               # [N_triplets]
        )  # [n_atoms, n_desc_angular, l_max]

        return {
            "g_radial": g_radial,
            "g_angular": g_angular
        }