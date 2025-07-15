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
        self.elements = para["elements"]
        self.n_elements = len(self.elements)
        
        input_dim = self.n_desc_radial + self.n_desc_angular * self.l_max
        hidden_dims = para["hidden_dims"]
        
        self.register_buffer('desc_scale', torch.ones(input_dim)) 
        self.register_buffer('desc_var_sum', torch.zeros(input_dim))  
        self.register_buffer('desc_count', torch.tensor(0.0)) 
        
        self.element_mlps = nn.ModuleDict()
        for _, element in enumerate(self.elements):
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.Tanh())
            for j in range(1, len(hidden_dims)):
                layers.append(nn.Linear(hidden_dims[j-1], hidden_dims[j]))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dims[-1], 1, bias=False))
            self.element_mlps[element] = nn.Sequential(*layers)
        
        self.shared_bias = nn.Parameter(torch.zeros(1))
    
    def update_descriptor_statistics(self, g_total):
        batch_size = g_total.shape[0]
        batch_var = g_total.var(dim=0, unbiased=False)

        self.desc_var_sum += batch_var * batch_size
        self.desc_count += batch_size
        if self.desc_count > 0:
            std = torch.sqrt(self.desc_var_sum / self.desc_count)
            self.desc_scale = 1.0 / torch.clamp(std, min=1e-8)  
        
    def normalize_descriptors(self, g_total):
        if self.training:
            self.update_descriptor_statistics(g_total)
        return g_total * self.desc_scale
    
    def forward(self, batch):
        positions = batch["positions"]
        types = batch["types"] 
        
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
        g_total = self.normalize_descriptors(g_total)

        e_atom = torch.zeros(n_atoms, device=g_total.device)
        for i, element in enumerate(self.elements):
            mask = (types == i)  
            if mask.any():
                g_element = g_total[mask]  
                e_element = self.element_mlps[element](g_element).squeeze(-1)  # [N_atoms_element]
                e_element = e_element + self.shared_bias
                e_atom[mask] = e_element
        
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
      
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        
        element_params = {}
        for element in self.elements:
            element_params[element] = sum(p.numel() for p in self.element_mlps[element].parameters())

        descriptor_params = sum(p.numel() for p in self.descriptor.parameters())
        radial_c_params = 0
        angular_c_params = 0
        if hasattr(self.descriptor, 'radial') and hasattr(self.descriptor.radial, 'c_table'):
            radial_c_params = self.descriptor.radial.c_table.numel()
        if hasattr(self.descriptor, 'angular') and hasattr(self.descriptor.angular, 'c_table'):
            angular_c_params = self.descriptor.angular.c_table.numel()
        
        input_dim = self.n_desc_radial + self.n_desc_angular * self.l_max
        
        print(f"总参数数量: {total_params}")
        print(f"描述符参数数量: {descriptor_params}")
        print(f"  - 径向c_table: {radial_c_params}")
        print(f"  - 角度c_table: {angular_c_params}")
        print(f"描述符数量信息:")
        print(f"  - 径向描述符: {self.n_desc_radial}")
        print(f"  - 角度描述符: {self.n_desc_angular} × {self.l_max} = {self.n_desc_angular * self.l_max}")
        print(f"  - 总描述符维度: {input_dim}")
        print("每个元素MLP的参数数量:")
        for element, count in element_params.items():
            print(f"  {element}: {count}")
        
        return {
            'total_params': total_params,
            'descriptor_params': descriptor_params,
            'radial_c_params': radial_c_params,
            'angular_c_params': angular_c_params,
            'element_params': element_params,
            'n_desc_radial': self.n_desc_radial,
            'n_desc_angular': self.n_desc_angular,
            'l_max': self.l_max,
            'total_descriptor_dim': input_dim
        }
    
    def save_to_nep_format(self, filepath):
        with open(filepath, 'w') as f:
            f.write(f"nep4 {len(self.elements)} " + " ".join(self.elements) + "\n")
            f.write(f"cutoff {self.para['rcut_radial']} {self.para['rcut_angular']} {self.para['NN_radial']} {self.para['NN_angular']}\n")
            f.write(f"n_max {self.para["n_desc_radial"]-1} {self.para["n_desc_angular"]-1}\n")
            f.write(f"basis_size {self.para["k_max_radial"]-1} {self.para["k_max_angular"]-1}\n")
            f.write(f"l_max {self.para["l_max"]}\n")
            f.write(f"ANN {self.para["hidden_dims"][0]} 0\n")

            for element in self.elements:
                element_params = []
                for param in self.element_mlps[element].parameters():
                    element_params.extend(param.data.flatten().tolist())
                for param_value in element_params:
                    f.write(f"{param_value:15.7e}\n")

            bias_value = self.shared_bias.data.item()
            f.write(f"{bias_value:15.7e}\n")

            radial_c_params = self.descriptor.radial.c_table.data.flatten()
            for param_value in radial_c_params:
                f.write(f"{param_value:15.7e}\n")

            angular_c_params = self.descriptor.angular.c_table.data.flatten()
            for param_value in angular_c_params:
                f.write(f"{param_value:15.7e}\n")

            input_dim = self.n_desc_radial + self.n_desc_angular * self.l_max
            for d in range(input_dim):
                scale_factor = self.desc_scale[d].item()  
                f.write(f"{scale_factor:15.7e}\n")



