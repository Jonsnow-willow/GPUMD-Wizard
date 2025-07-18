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

        e_atom = torch.zeros(n_atoms, device=g_total.device)
        for i, element in enumerate(self.elements):
            mask = (types == i)  
            if mask.any():
                g_element = g_total[mask]  
                e_element = self.element_mlps[element](g_element).squeeze(-1)  # [N_atoms_element]
                e_element -= self.shared_bias
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
    
    @classmethod
    def from_checkpoint(cls, filepath, device=None):
        checkpoint = torch.load(filepath, map_location=device)
        para = checkpoint['para']
        model = cls(para)
        model.load_state_dict(checkpoint['model_state_dict'])

        if device is not None:
            model.to(device)
        
        return model
      
    def print_model_info(self):
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
        
        print(f"Total number of parameters: {total_params}")
        print(f"Number of descriptor parameters: {descriptor_params}")
        print(f"  - Radial c_table: {radial_c_params}")
        print(f"  - Angular c_table: {angular_c_params}")
        print(f"Descriptor dimension information:")
        print(f"  - Radial descriptors: {self.n_desc_radial}")
        print(f"  - Angular descriptors: {self.n_desc_angular} × {self.l_max} = {self.n_desc_angular * self.l_max}")
        print(f"  - Total descriptor dimension: {input_dim}")
        print("Number of parameters in each element MLP:")
        for element, count in element_params.items():
            print(f"  {element}: {count}")
            
    def save_to_nep_format(self, filepath):
        with open(filepath, 'w') as f:
            f.write(f"nep4 {len(self.elements)} " + " ".join(self.elements) + "\n")
            f.write(f"cutoff {self.para['rcut_radial']} {self.para['rcut_angular']} {self.para['NN_radial']} {self.para['NN_angular']}\n")
            f.write(f"n_max {int(self.para['n_desc_radial']) - 1} {int(self.para['n_desc_angular']) - 1}\n")
            f.write(f"basis_size {int(self.para['k_max_radial']) - 1} {int(self.para['k_max_angular']) - 1}\n")
            f.write(f"l_max {int(self.para['l_max'])} 0 0\n")
            f.write(f"ANN {int(self.para['hidden_dims'][0])} 0\n")

            for element in self.elements:
                mlp = self.element_mlps[element]
                for idx, layer in enumerate(mlp):
                    if isinstance(layer, nn.Linear):
                        weights = layer.weight.data.flatten().tolist()
                        for val in weights:
                            f.write(f"{val:15.7e}\n")

                        if layer.bias is not None:
                            bias = layer.bias.data
                            if idx == 0:  
                                bias = -bias
                            bias = bias.flatten().tolist()
                            for val in bias:
                                f.write(f"{val:15.7e}\n")

            bias_value = self.shared_bias.data.item()
            f.write(f"{bias_value:15.7e}\n")

            n_max_radial = int(self.para['n_desc_radial'])
            n_max_angular = int(self.para['n_desc_angular'])
            k_max_radial = int(self.para['k_max_radial'])
            k_max_angular = int(self.para['k_max_angular'])
            n_types = len(self.elements)

            radial_params = self.descriptor.radial.c_table.data
            for n in range(n_max_radial):
                for k in range(k_max_radial):
                    for t1 in range(n_types):
                        for t2 in range(n_types):
                            val = radial_params[t1, t2, n, k].item()
                            f.write(f"{val:15.7e}\n")

            angular_params = self.descriptor.angular.c_table.data
            for n in range(n_max_angular):
                for k in range(k_max_angular):
                    for t1 in range(n_types):
                        for t2 in range(n_types):
                            val = angular_params[t1, t2, n, k].item()
                            f.write(f"{val:15.7e}\n")

            input_dim = self.n_desc_radial + self.n_desc_angular * self.l_max
            for _ in range(input_dim):
                f.write(f"{1.0:15.7e}\n")



