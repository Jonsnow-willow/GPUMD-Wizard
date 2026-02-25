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
        self.L_max = para["l_max"]
        self.l_max_4body = para.get("l_max_4body", 2)
        self.l_max_5body = para.get("l_max_5body", 0)
        self.num_L = self.descriptor.angular.num_L
        self.elements = para["elements"]
        self.n_elements = len(self.elements)
        
        input_dim = self.n_desc_radial + self.n_desc_angular * self.num_L
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
        
        self._init_descriptor_scaler(input_dim)
        self.shared_bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        positions = batch["positions"]
        types = batch["types"] 
        
        if not positions.requires_grad:
            positions = positions.clone().detach().requires_grad_(True)
            batch = batch.copy()
            batch["positions"] = positions
        
        descriptors = self.descriptor(batch)
        g_total = self._combine_descriptors(descriptors)         # [N_atoms, input_dim]
        n_atoms = g_total.shape[0]
        g_total = g_total * self.q_scaler

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
        virial_list = []

        forces = None
        if positions.requires_grad:
            forces = -torch.autograd.grad(
                outputs=e_atom.sum(),
                inputs=positions,
                grad_outputs=torch.ones_like(e_atom.sum()),
                create_graph=self.training,
                retain_graph=True
            )[0]

        for n_atoms_in_structure in n_atoms_per_structure:
            e_structure = e_atom[atom_idx:atom_idx + n_atoms_in_structure].sum()
            e_total_list.append(e_structure)
            if positions.requires_grad and forces is not None:
                pos_struct = positions[atom_idx:atom_idx + n_atoms_in_structure]
                force_struct = forces[atom_idx:atom_idx + n_atoms_in_structure]
                virial_struct = -torch.einsum('ni,nj->ij', pos_struct, force_struct)
                virial_list.append(virial_struct.reshape(-1))
            atom_idx += n_atoms_in_structure

        e_total = torch.stack(e_total_list)
        virial = torch.stack(virial_list) if virial_list else None

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
        state = checkpoint.get('model_state_dict', {})
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys and "q_scaler" in incompatible.missing_keys:
            scaler = para.get("descriptor_scaler", None)
            if scaler is not None and len(scaler) == model.q_scaler.numel():
                with torch.no_grad():
                    model.q_scaler.copy_(torch.tensor(scaler, dtype=model.q_scaler.dtype))

        if device is not None:
            model.to(device)
        
        return model

    @classmethod
    def from_nep_txt(cls, filepath, device=None):
        def float_stream(lines, start):
            for line in lines[start:]:
                for token in line.split():
                    yield float(token)

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip() != ""]

        idx = 0
        tokens = lines[idx].split()
        idx += 1
        if not tokens[0].startswith("nep"):
            raise ValueError("Invalid nep.txt: first line should start with nep4 or nep5.")
        version = int(tokens[0][3:])
        if version != 4:
            raise NotImplementedError("Only nep4 is supported by torchNEP loader.")
        num_types = int(tokens[1])
        elements = tokens[2:]
        if len(elements) != num_types:
            raise ValueError("Invalid nep.txt: element count mismatch.")

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "cutoff":
            raise ValueError("Invalid nep.txt: cutoff line not found.")
        if len(tokens) == 5:
            rcut_radial = float(tokens[1])
            rcut_angular = float(tokens[2])
            NN_radial = int(tokens[3])
            NN_angular = int(tokens[4])
        else:
            # cutoff rc_r1 rc_a1 rc_r2 rc_a2 ... MN_radial MN_angular
            expected = num_types * 2 + 3
            if len(tokens) != expected:
                raise ValueError("Invalid nep.txt: cutoff line length mismatch.")
            rc_radial = [float(tokens[1 + i * 2]) for i in range(num_types)]
            rc_angular = [float(tokens[2 + i * 2]) for i in range(num_types)]
            if len(set(rc_radial)) != 1 or len(set(rc_angular)) != 1:
                raise NotImplementedError("Per-type cutoffs are not supported in torchNEP.")
            rcut_radial = rc_radial[0]
            rcut_angular = rc_angular[0]
            NN_radial = int(tokens[-2])
            NN_angular = int(tokens[-1])

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "n_max":
            raise ValueError("Invalid nep.txt: n_max line not found.")
        n_max_radial = int(tokens[1])
        n_max_angular = int(tokens[2])

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "basis_size":
            raise ValueError("Invalid nep.txt: basis_size line not found.")
        basis_size_radial = int(tokens[1])
        basis_size_angular = int(tokens[2])

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "l_max":
            raise ValueError("Invalid nep.txt: l_max line not found.")
        l_max = int(tokens[1])
        l_max_4body = int(tokens[2])
        l_max_5body = int(tokens[3])

        tokens = lines[idx].split()
        idx += 1
        if tokens[0] != "ANN":
            raise ValueError("Invalid nep.txt: ANN line not found.")
        num_neurons = int(tokens[1])

        n_desc_radial = n_max_radial + 1
        n_desc_angular = n_max_angular + 1
        k_max_radial = basis_size_radial + 1
        k_max_angular = basis_size_angular + 1

        para = {
            "elements": elements,
            "rcut_radial": rcut_radial,
            "rcut_angular": rcut_angular,
            "n_desc_radial": n_desc_radial,
            "n_desc_angular": n_desc_angular,
            "k_max_radial": k_max_radial,
            "k_max_angular": k_max_angular,
            "l_max": l_max,
            "l_max_4body": l_max_4body,
            "l_max_5body": l_max_5body,
            "NN_radial": NN_radial,
            "NN_angular": NN_angular,
            "hidden_dims": [num_neurons],
            "n_types": num_types,
        }

        model = cls(para)
        num_L = model.num_L
        dim = n_desc_radial + n_desc_angular * num_L

        stream = float_stream(lines, idx)

        with torch.no_grad():
            for t, element in enumerate(elements):
                w0 = torch.tensor([next(stream) for _ in range(num_neurons * dim)],
                                  dtype=torch.float32).reshape(num_neurons, dim)
                b0 = torch.tensor([next(stream) for _ in range(num_neurons)], dtype=torch.float32)
                w1 = torch.tensor([next(stream) for _ in range(num_neurons)], dtype=torch.float32)

                mlp = model.element_mlps[element]
                mlp[0].weight.copy_(w0)
                mlp[0].bias.copy_(-b0)
                mlp[-1].weight.copy_(w1.view(1, -1))

            shared_bias = torch.tensor(next(stream), dtype=torch.float32)
            model.shared_bias.copy_(shared_bias)

            c_radial = torch.empty(num_types, num_types, n_desc_radial, k_max_radial, dtype=torch.float32)
            for n in range(n_desc_radial):
                for k in range(k_max_radial):
                    for t1 in range(num_types):
                        for t2 in range(num_types):
                            c_radial[t1, t2, n, k] = next(stream)
            model.descriptor.radial.c_table.copy_(c_radial)

            c_angular = torch.empty(num_types, num_types, n_desc_angular, k_max_angular, dtype=torch.float32)
            for n in range(n_desc_angular):
                for k in range(k_max_angular):
                    for t1 in range(num_types):
                        for t2 in range(num_types):
                            c_angular[t1, t2, n, k] = next(stream)
            model.descriptor.angular.c_table.copy_(c_angular)

            q_scaler = torch.tensor([next(stream) for _ in range(dim)], dtype=torch.float32)
            model.q_scaler.copy_(q_scaler)
            model.para["descriptor_scaler"] = q_scaler.detach().cpu().tolist()

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
        
        input_dim = self.n_desc_radial + self.n_desc_angular * self.num_L
        
        print(f"Total number of parameters: {total_params}")
        print(f"Number of descriptor parameters: {descriptor_params}")
        print(f"  - Radial c_table: {radial_c_params}")
        print(f"  - Angular c_table: {angular_c_params}")
        print(f"Descriptor dimension information:")
        print(f"  - Radial descriptors: {self.n_desc_radial}")
        print(f"  - Angular descriptors: {self.n_desc_angular} × {self.num_L} = {self.n_desc_angular * self.num_L}")
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
            f.write(f"l_max {int(self.L_max)} {int(self.l_max_4body)} {int(self.l_max_5body)}\n")
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

            scaler = self.q_scaler.detach().cpu().reshape(-1)
            for val in scaler.tolist():
                f.write(f"{val:15.7e}\n")

    def _init_descriptor_scaler(self, input_dim):
        scaler_tensor = torch.ones(input_dim, dtype=torch.float32)
        self.register_buffer("q_scaler", scaler_tensor)
        self.para["descriptor_scaler"] = scaler_tensor.tolist()

    def compute_descriptor_scaler(self, dataloader, device=None):
        """
        Estimate scaler = 1 / (max - min) for each descriptor dimension.
        """
        if device is None:
            device = next(self.parameters()).device
        prev_mode = self.training
        self.eval()
        q_min = None
        q_max = None
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                descriptors = self.descriptor(batch)
                g_total = self._combine_descriptors(descriptors)
                if g_total.numel() == 0:
                    continue
                batch_min = torch.amin(g_total, dim=0)
                batch_max = torch.amax(g_total, dim=0)
                if q_min is None:
                    q_min = batch_min
                    q_max = batch_max
                else:
                    q_min = torch.minimum(q_min, batch_min)
                    q_max = torch.maximum(q_max, batch_max)
        if q_min is not None:
            diff = q_max - q_min
            scaler = torch.ones_like(diff)
            valid = diff > 1.0e-12
            scaler[valid] = 1.0 / diff[valid]
            tensor = scaler.flatten()
            with torch.no_grad():
                self.q_scaler.copy_(tensor)
            self.para["descriptor_scaler"] = tensor.detach().cpu().tolist()
        self.train(prev_mode)
        return self.q_scaler.detach().cpu().clone()

    def _combine_descriptors(self, descriptors):
        g_radial = descriptors["g_radial"]
        g_angular = descriptors["g_angular"]
        n_atoms = g_radial.shape[0]
        if n_atoms == 0:
            return g_radial.new_zeros((0, self.n_desc_radial + self.n_desc_angular * self.num_L))
        g_angular_flat = g_angular.permute(0, 2, 1).reshape(n_atoms, -1)
        return torch.cat([g_radial, g_angular_flat], dim=-1)
