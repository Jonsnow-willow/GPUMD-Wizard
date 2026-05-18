import re
import torch
import torch.nn as nn
from ase.data import atomic_numbers
from .descriptor import Descriptor 

K_C_SP = 14.399645
ZBL_A_INV_FACTOR = 2.134563
UNIVERSAL_ZBL_COEFFS = [
    0.18175, 3.1998,
    0.50986, 0.94229,
    0.28022, 0.4029,
    0.02817, 0.20162,
]

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
        self._configure_zbl()
        self.shared_bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        positions = batch["positions"]
        types = batch["types"] 
        
        if not positions.requires_grad:
            positions = positions.clone().detach().requires_grad_(True)
            batch = batch.copy()
            batch["positions"] = positions
        else:
            batch = batch.copy()
        
        strain = None
        descriptor_batch = batch
        if batch.get("compute_virial", True):
            strain = torch.zeros((3, 3), device=positions.device, dtype=positions.dtype, requires_grad=True)
            deformation = torch.eye(3, device=positions.device, dtype=positions.dtype) + strain
            descriptor_batch = batch.copy()
            descriptor_batch["positions"] = positions @ deformation.T
            descriptor_batch["radial_offsets"] = batch["radial_offsets"].to(device=positions.device, dtype=positions.dtype) @ deformation.T
            descriptor_batch["angular_offsets"] = batch["angular_offsets"].to(device=positions.device, dtype=positions.dtype) @ deformation.T
        
        descriptors = self.descriptor(descriptor_batch)
        g_total = self._combine_descriptors(descriptors)         # [N_atoms, input_dim]
        n_atoms = g_total.shape[0]
        g_total = g_total * self.q_scaler

        e_atom = torch.zeros(n_atoms, device=g_total.device, dtype=g_total.dtype)
        for i, element in enumerate(self.elements):
            mask = (types == i)  
            if mask.any():
                g_element = g_total[mask]  
                e_element = self.element_mlps[element](g_element).squeeze(-1)  # [N_atoms_element]
                e_element -= self.shared_bias
                e_atom[mask] = e_element

        if self.zbl_enabled:
            e_atom = e_atom + self._zbl_atomic_energy(
                types=types,
                positions=descriptor_batch["positions"],
                angular_neighbors=descriptor_batch["angular_neighbors"],
                angular_offsets=descriptor_batch["angular_offsets"],
            )
        
        n_atoms_per_structure = batch["n_atoms_per_structure"].detach().cpu().tolist()
        atom_idx = 0
        e_total_list = []
        virial_list = []

        forces = None
        total_energy = e_atom.sum()
        if positions.requires_grad:
            forces = -torch.autograd.grad(
                outputs=total_energy,
                inputs=positions,
                create_graph=self.training,
                retain_graph=True
            )[0]

        for n_atoms_in_structure in n_atoms_per_structure:
            e_structure = e_atom[atom_idx:atom_idx + n_atoms_in_structure].sum()
            e_total_list.append(e_structure)
            if strain is not None:
                virial_struct = torch.autograd.grad(
                    outputs=e_structure,
                    inputs=strain,
                    create_graph=self.training,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if virial_struct is None:
                    virial_struct = torch.zeros((3, 3), device=g_total.device, dtype=g_total.dtype)
                virial_list.append((-virial_struct).reshape(-1))
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
        header = tokens[0].lower()
        match = re.fullmatch(r"nep(\d+)(?:_(.*))?", header)
        if match is None:
            raise ValueError("Invalid nep.txt: first line should start with nep4 or nep5.")
        version = int(match.group(1))
        suffix = match.group(2) or ""
        has_zbl = "zbl" in suffix.split("_")
        if version != 4:
            raise NotImplementedError("Only nep4 is supported by torchNEP loader.")
        num_types = int(tokens[1])
        elements = tokens[2:]
        if len(elements) != num_types:
            raise ValueError("Invalid nep.txt: element count mismatch.")

        zbl = None
        if has_zbl:
            tokens = lines[idx].split()
            idx += 1
            if tokens[0] != "zbl" or len(tokens) not in (3, 4):
                raise ValueError("Invalid nep.txt: zbl line not found after nep4_zbl header.")
            zbl = {
                "rc_inner": float(tokens[1]),
                "rc_outer": float(tokens[2]),
                "flexible": float(tokens[1]) == 0.0 and float(tokens[2]) == 0.0,
            }
            if len(tokens) == 4:
                zbl["typewise_cutoff_factor"] = float(tokens[3])
                raise NotImplementedError("Typewise ZBL cutoffs are not supported in torchNEP.")

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
        if zbl is not None and not zbl["flexible"]:
            para["zbl"] = zbl

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

            if zbl is not None and zbl["flexible"]:
                num_type_zbl = num_types * (num_types + 1) // 2
                zbl_parameters = torch.tensor(
                    [next(stream) for _ in range(10 * num_type_zbl)],
                    dtype=torch.float32,
                ).reshape(num_type_zbl, 10)
                zbl["parameters"] = zbl_parameters.detach().cpu().reshape(-1).tolist()
                model.para["zbl"] = zbl
                model._configure_zbl()

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
            zbl = self.para.get("zbl", None)
            header = "nep4_zbl" if zbl is not None else "nep4"
            f.write(f"{header} {len(self.elements)} " + " ".join(self.elements) + "\n")
            if zbl is not None:
                zbl_config = self._normalize_zbl_config(zbl)
                f.write(f"zbl {zbl_config['rc_inner']} {zbl_config['rc_outer']}\n")
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

            if zbl is not None:
                zbl_config = self._normalize_zbl_config(zbl)
                if zbl_config["flexible"]:
                    for val in zbl_config["parameters"]:
                        f.write(f"{val:15.7e}\n")

    def _init_descriptor_scaler(self, input_dim):
        scaler_tensor = torch.ones(input_dim, dtype=torch.float32)
        self.register_buffer("q_scaler", scaler_tensor)
        self.para["descriptor_scaler"] = scaler_tensor.tolist()

    def _normalize_zbl_config(self, zbl):
        if zbl is None:
            return None
        if isinstance(zbl, dict):
            rc_inner = float(zbl["rc_inner"])
            rc_outer = float(zbl["rc_outer"])
            flexible = bool(zbl.get("flexible", rc_inner == 0.0 and rc_outer == 0.0))
            parameters = zbl.get("parameters")
            if flexible and parameters is None:
                raise ValueError("Flexible ZBL requires fitted ZBL parameters.")
            return {
                "rc_inner": rc_inner,
                "rc_outer": rc_outer,
                "flexible": flexible,
                "parameters": parameters,
            }
        if len(zbl) != 2:
            raise ValueError("ZBL config should be a dict or [rc_inner, rc_outer].")
        rc_inner = float(zbl[0])
        rc_outer = float(zbl[1])
        flexible = rc_inner == 0.0 and rc_outer == 0.0
        if flexible:
            raise ValueError("Flexible ZBL requires fitted ZBL parameters.")
        return {
            "rc_inner": rc_inner,
            "rc_outer": rc_outer,
            "flexible": False,
            "parameters": None,
        }

    def _configure_zbl(self):
        zbl = self._normalize_zbl_config(self.para.get("zbl", None))
        self.zbl_enabled = zbl is not None
        self.zbl_config = zbl

        atomic_number_tensor = torch.tensor(
            [atomic_numbers[element] for element in self.elements],
            dtype=torch.float32,
        )
        if hasattr(self, "zbl_atomic_numbers"):
            self.zbl_atomic_numbers = atomic_number_tensor.to(self.zbl_atomic_numbers.device)
        else:
            self.register_buffer("zbl_atomic_numbers", atomic_number_tensor)

        if zbl is not None and zbl["flexible"]:
            params = torch.tensor(zbl["parameters"], dtype=torch.float32).reshape(-1, 10)
        else:
            params = torch.zeros((0, 10), dtype=torch.float32)
        if hasattr(self, "zbl_parameters"):
            self.zbl_parameters = params.to(self.zbl_parameters.device)
        else:
            self.register_buffer("zbl_parameters", params)

        coeffs = torch.tensor(UNIVERSAL_ZBL_COEFFS, dtype=torch.float32).reshape(4, 2)
        if hasattr(self, "universal_zbl_coeffs"):
            self.universal_zbl_coeffs = coeffs.to(self.universal_zbl_coeffs.device)
        else:
            self.register_buffer("universal_zbl_coeffs", coeffs)

    def _zbl_atomic_energy(self, types, positions, angular_neighbors, angular_offsets):
        mask = angular_neighbors != -1
        n_atoms = positions.shape[0]
        if not mask.any():
            return positions.new_zeros(n_atoms)

        safe_neighbors = angular_neighbors.clone()
        safe_neighbors[~mask] = 0
        r_vec = (
            positions[safe_neighbors]
            + angular_offsets.to(device=positions.device, dtype=positions.dtype)
            - positions.unsqueeze(1)
        )
        distances = torch.linalg.norm(r_vec, dim=-1)
        safe_distances = torch.where(mask, torch.clamp(distances, min=1.0e-12), torch.ones_like(distances))
        type_i = types.unsqueeze(1).expand_as(safe_neighbors)
        type_j = types[safe_neighbors]
        zbl_atomic_numbers = self.zbl_atomic_numbers.to(device=positions.device, dtype=positions.dtype)
        z_i = zbl_atomic_numbers[type_i]
        z_j = zbl_atomic_numbers[type_j]
        a_inv = (torch.pow(z_i, 0.23) + torch.pow(z_j, 0.23)) * ZBL_A_INV_FACTOR
        zizj = K_C_SP * z_i * z_j

        if self.zbl_config["flexible"]:
            t1 = torch.minimum(type_i, type_j)
            t2 = torch.maximum(type_i, type_j)
            zbl_index = t1 * self.n_elements - (t1 * (t1 - 1)) // 2 + (t2 - t1)
            params = self.zbl_parameters.to(device=positions.device, dtype=positions.dtype)[zbl_index]
            rc_inner = params[..., 0]
            rc_outer = params[..., 1]
            coeffs = params[..., 2:].reshape(*params.shape[:-1], 4, 2)
        else:
            rc_inner = positions.new_full(distances.shape, self.zbl_config["rc_inner"])
            rc_outer = positions.new_full(distances.shape, self.zbl_config["rc_outer"])
            coeffs = self.universal_zbl_coeffs.to(device=positions.device, dtype=positions.dtype).view(1, 1, 4, 2)

        x = safe_distances * a_inv
        phi = torch.sum(coeffs[..., 0] * torch.exp(-coeffs[..., 1] * x.unsqueeze(-1)), dim=-1)
        screened = zizj * phi / safe_distances
        cutoff_width = torch.clamp(rc_outer - rc_inner, min=1.0e-12)
        fc = torch.where(
            safe_distances < rc_inner,
            torch.ones_like(safe_distances),
            torch.where(
                safe_distances < rc_outer,
                0.5 * torch.cos(torch.pi * (safe_distances - rc_inner) / cutoff_width) + 0.5,
                torch.zeros_like(safe_distances),
            ),
        )
        edge_energy = 0.5 * screened * fc * mask
        return torch.sum(edge_energy, dim=1)

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
