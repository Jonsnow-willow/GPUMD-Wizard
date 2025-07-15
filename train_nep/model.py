import torch
import torch.nn as nn
from .descriptor import Descriptor 

class NEP(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.descriptor = Descriptor(para)
        
        self.n_desc_radial = para["n_desc_radial"]
        self.n_desc_angular = para["n_desc_angular"]
        self.l_max = para["l_max"]
        
        input_dim = self.n_desc_radial + self.n_desc_angular * self.l_max
        
        hidden_dims = para["hidden_dims"]
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.Tanh())
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, batch):
        positions = batch["positions"]
        if not positions.requires_grad:
            positions = positions.clone().detach().requires_grad_(True)
            batch = batch.copy()
            batch["positions"] = positions
        
        descriptors = self.descriptor(batch)
        g_radial = descriptors["g_radial"]                       # [N_atoms, n_desc_radial]
        g_angular = descriptors["g_angular"]                     # [N_atoms, n_desc_angular, l_max]
        
        n_atoms = g_radial.shape[0]
        g_angular_flat = g_angular.reshape(n_atoms, -1)          # [N_atoms, n_desc_angular * l_max]
        g_total = torch.cat([g_radial, g_angular_flat], dim=-1)  # [N_atoms, input_dim]

        e_atom = self.mlp(g_total).squeeze(-1)                   # [N_atoms]
        
        n_atoms_per_structure = batch["n_atoms_per_structure"]
        atom_idx = 0
        e_total_list = []
        for n_atoms_in_structure in n_atoms_per_structure:
            e_structure = e_atom[atom_idx:atom_idx + n_atoms_in_structure].sum()
            e_total_list.append(e_structure)
            atom_idx += n_atoms_in_structure
        e_total = torch.stack(e_total_list)  
        
        forces = None
        virial = None
        if positions.requires_grad:
            forces = -torch.autograd.grad(
                outputs=e_total.sum(),
                inputs=positions,
                grad_outputs=torch.ones_like(e_total.sum()),
                create_graph=self.training,
                retain_graph=True
            )[0]
            virial = -torch.einsum('ni,nj->ij', positions, forces)
        
        result = {
            "energies": e_atom,          
            "energy": e_total,         
        }
        if forces is not None:
            result["forces"] = forces        
        if virial is not None:
            result["virial"] = virial        
        return result