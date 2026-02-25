import torch
import torch.nn as nn

NUM_OF_ABC = 80  # 3 + 5 + ... + 17 for L_max = 8
C3B_VALUES = [
    0.238732414637843,
    0.119366207318922,
    0.119366207318922,
    0.099471839432435,
    0.596831036594608,
    0.596831036594608,
    0.149207759148652,
    0.149207759148652,
    0.139260575205408,
    0.104445431404056,
    0.104445431404056,
    1.044454314040563,
    1.044454314040563,
    0.174075719006761,
    0.174075719006761,
    0.011190581936149,
    0.223811638722978,
    0.223811638722978,
    0.111905819361489,
    0.111905819361489,
    1.566681471060845,
    1.566681471060845,
    0.195835183882606,
    0.195835183882606,
    0.01367737792196,
    0.102580334414698,
    0.102580334414698,
    2.872249363611549,
    2.872249363611549,
    0.119677056817148,
    0.119677056817148,
    2.154187022708661,
    2.154187022708661,
    0.215418702270866,
    0.215418702270866,
    0.004041043476943,
    0.169723826031592,
    0.169723826031592,
    0.106077391269745,
    0.106077391269745,
    0.424309565078979,
    0.424309565078979,
    0.127292869523694,
    0.127292869523694,
    2.80044312952126,
    2.80044312952126,
    0.233370260793438,
    0.233370260793438,
    0.004662742473395,
    0.004079899664221,
    0.004079899664221,
    0.024479397985326,
    0.024479397985326,
    0.012239698992663,
    0.012239698992663,
    0.538546755677165,
    0.538546755677165,
    0.134636688919291,
    0.134636688919291,
    3.500553911901575,
    3.500553911901575,
    0.250039565135827,
    0.250039565135827,
    8.2569397966e-05,
    0.005944996653579,
    0.005944996653579,
    0.104037441437634,
    0.104037441437634,
    0.762941237209318,
    0.762941237209318,
    0.114441185581398,
    0.114441185581398,
    5.950941650232678,
    5.950941650232678,
    0.141689086910302,
    0.141689086910302,
    4.250672607309055,
    4.250672607309055,
    0.265667037956816,
    0.265667037956816,
]
C4B_VALUES = [
    -0.007499480826664,
    -0.134990654879954,
    0.067495327439977,
    0.404971964639861,
    -0.809943929279723,
]
C5B_VALUES = [0.026596810706114, 0.053193621412227, 0.026596810706114]
Z_COEFF_DATA = {
    1: [[0.0, 1.0], [1.0, 0.0]],
    2: [[-1.0, 0.0, 3.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    3: [
        [0.0, -3.0, 0.0, 5.0],
        [-1.0, 0.0, 5.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ],
    4: [
        [3.0, 0.0, -30.0, 0.0, 35.0],
        [0.0, -3.0, 0.0, 7.0, 0.0],
        [-1.0, 0.0, 7.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    5: [
        [0.0, 15.0, 0.0, -70.0, 0.0, 63.0],
        [1.0, 0.0, -14.0, 0.0, 21.0, 0.0],
        [0.0, -1.0, 0.0, 3.0, 0.0, 0.0],
        [-1.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    6: [
        [-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0],
        [0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0],
        [1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0],
        [0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    7: [
        [0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0],
        [-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0],
        [0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0],
        [3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0],
        [0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    8: [
        [35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0],
        [0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0],
        [-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
}


def chebyshev_basis(r, r_c, k_max):
    """
    Evaluate the modified Chebyshev basis used in NEP.
    Works for arbitrary-shaped tensors `r` (broadcasted element-wise).
    """
    fc = torch.where(r < r_c, 0.5 * torch.cos(torch.pi * r / r_c) + 0.5, torch.zeros_like(r))
    x = 2.0 * (r / r_c - 1.0) ** 2 - 1.0
    half_fc = 0.5 * fc
    fn = [torch.ones_like(x)]
    if k_max > 1:
        fn.append(x)
        for _ in range(2, k_max):
            fn.append(2.0 * x * fn[-1] - fn[-2])
    fn = torch.stack(fn, dim=-1)
    fn = (fn + 1.0) * half_fc.unsqueeze(-1)
    return fn


class RadialDescriptor(nn.Module):
    def __init__(self, n_types, n_desc, k_max, r_c):
        super().__init__()
        self.n_types = n_types
        self.n_desc = n_desc
        self.k_max = k_max
        self.r_c = r_c
        self.c_table = nn.Parameter(torch.randn(n_types, n_types, n_desc, k_max))

    def forward(self, types, positions, radial_neighbors, radial_offsets):
        n_atoms, nn_radial = radial_neighbors.shape
        mask = radial_neighbors != -1
        if not mask.any():
            return positions.new_zeros((n_atoms, self.n_desc))

        safe_neighbors = radial_neighbors.clone()
        safe_neighbors[~mask] = 0

        pos_i = positions.unsqueeze(1)
        pos_j = positions[safe_neighbors]
        r_vec = pos_j + radial_offsets.to(dtype=positions.dtype) - pos_i
        distances = torch.linalg.norm(r_vec, dim=-1)

        f = chebyshev_basis(distances, self.r_c, self.k_max)
        type_i = types.unsqueeze(1).expand(-1, nn_radial)
        type_j = types[safe_neighbors]
        coeff = self.c_table[type_i, type_j]
        edge_desc = torch.sum(coeff * f.unsqueeze(2), dim=-1)
        edge_desc = edge_desc * mask.unsqueeze(-1)

        g = torch.sum(edge_desc, dim=1)
        return g


class AngularDescriptor(nn.Module):
    def __init__(self, n_types, n_desc, k_max, r_c, l_max, l_max_4body=2, l_max_5body=0):
        super().__init__()
        self.n_types = n_types
        self.n_desc = n_desc
        self.k_max = k_max
        self.r_c = r_c
        self.L_max = l_max
        self.include_4body = l_max_4body == 2
        self.include_5body = l_max_5body == 1
        self.num_L = self.L_max + int(self.include_4body) + int(self.include_5body)

        self.c_table = nn.Parameter(torch.randn(n_types, n_types, n_desc, k_max))
        self.register_buffer("C3B", torch.tensor(C3B_VALUES, dtype=torch.float32))
        self.register_buffer("C4B", torch.tensor(C4B_VALUES, dtype=torch.float32))
        self.register_buffer("C5B", torch.tensor(C5B_VALUES, dtype=torch.float32))
        for L, data in Z_COEFF_DATA.items():
            tensor = torch.tensor(data, dtype=torch.float32)
            self.register_buffer(f"Z{L}", tensor)

    def _gather_neighbor_vectors(self, positions, neighbors, neighbor_offsets):
        mask = neighbors != -1
        if not mask.any():
            return None, None, None
        safe_index = neighbors.clone()
        safe_index[~mask] = 0
        pos_i = positions.unsqueeze(1)
        pos_j = positions[safe_index]
        r_vec = pos_j + neighbor_offsets.to(dtype=positions.dtype) - pos_i
        r_vec = torch.where(mask.unsqueeze(-1), r_vec, torch.zeros_like(r_vec))
        distances = torch.linalg.norm(r_vec, dim=-1)
        safe_valid = torch.clamp(distances, min=1.0e-12)
        safe_dist = torch.where(mask, safe_valid, torch.ones_like(distances))
        unit_vec = torch.where(mask.unsqueeze(-1), r_vec / safe_dist.unsqueeze(-1), torch.zeros_like(r_vec))
        return unit_vec, distances, mask

    def _compute_basis(self, unit_vec, mask_flat):
        if self.L_max == 0 or unit_vec.numel() == 0:
            return unit_vec.new_zeros((unit_vec.shape[0], NUM_OF_ABC))

        x = unit_vec[:, 0]
        y = unit_vec[:, 1]
        z = unit_vec[:, 2]
        m = unit_vec.shape[0]
        device = unit_vec.device
        dtype = unit_vec.dtype

        z_pow = [torch.ones(m, device=device, dtype=dtype)]
        for _ in range(1, self.L_max + 1):
            z_pow.append(z_pow[-1] * z)
        z_pow = torch.stack(z_pow, dim=0)

        xy_real = [torch.ones(m, device=device, dtype=dtype)]
        xy_imag = [torch.zeros(m, device=device, dtype=dtype)]
        for _ in range(1, self.L_max + 1):
            real_prev = xy_real[-1]
            imag_prev = xy_imag[-1]
            xy_real.append(real_prev * x - imag_prev * y)
            xy_imag.append(real_prev * y + imag_prev * x)
        xy_real = torch.stack(xy_real, dim=0)
        xy_imag = torch.stack(xy_imag, dim=0)

        basis = torch.zeros((m, NUM_OF_ABC), device=device, dtype=dtype)
        for L in range(1, self.L_max + 1):
            z_coeff = getattr(self, f"Z{L}")
            start = L * L - 1
            idx = start
            for n1 in range(L + 1):
                n2_start = 0 if (L + n1) % 2 == 0 else 1
                n2_indices = torch.arange(n2_start, L - n1 + 1, 2, device=device)
                if n2_indices.numel() == 0:
                    z_factor = torch.zeros_like(x)
                else:
                    coeffs = z_coeff[n1, n2_indices]
                    z_terms = z_pow[n2_indices]
                    z_factor = torch.sum(coeffs.unsqueeze(1) * z_terms, dim=0)
                if n1 == 0:
                    basis[:, idx] = z_factor
                    idx += 1
                else:
                    basis[:, idx] = z_factor * xy_real[n1]
                    basis[:, idx + 1] = z_factor * xy_imag[n1]
                    idx += 2

        if mask_flat is not None:
            basis = basis * mask_flat.unsqueeze(-1)
        return basis

    def _compute_q(self, s):
        q_list = []
        for L in range(1, self.L_max + 1):
            start = L * L - 1
            num_terms = 2 * L + 1
            coeffs = self.C3B[start : start + num_terms]
            comps = s[..., start : start + num_terms]
            q = coeffs[0] * comps[..., 0] * comps[..., 0]
            if num_terms > 1:
                q = q + 2.0 * torch.sum(coeffs[1:] * comps[..., 1:] * comps[..., 1:], dim=-1)
            q_list.append(q)

        if self.include_4body:
            if self.L_max < 2:
                raise ValueError("L_max must be >= 2 to include 4-body descriptors.")
            s3 = s[..., 3]
            s4 = s[..., 4]
            s5 = s[..., 5]
            s6 = s[..., 6]
            s7 = s[..., 7]
            q4 = (
                self.C4B[0] * s3 * s3 * s3
                + self.C4B[1] * s3 * (s4 * s4 + s5 * s5)
                + self.C4B[2] * s3 * (s6 * s6 + s7 * s7)
                + self.C4B[3] * s6 * (s5 * s5 - s4 * s4)
                + self.C4B[4] * s4 * s5 * s7
            )
            q_list.append(q4)

        if self.include_5body:
            if self.L_max < 1:
                raise ValueError("L_max must be >= 1 to include 5-body descriptors.")
            s0 = s[..., 0]
            s1 = s[..., 1]
            s2 = s[..., 2]
            s0_sq = s0 * s0
            s1_sq_plus_s2_sq = s1 * s1 + s2 * s2
            q5 = (
                self.C5B[0] * s0_sq * s0_sq
                + self.C5B[1] * s0_sq * s1_sq_plus_s2_sq
                + self.C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq
            )
            q_list.append(q5)

        return torch.stack(q_list, dim=-1) if q_list else s.new_zeros((s.shape[0], s.shape[1], 0))

    def forward(self, types, positions, angular_neighbors, angular_offsets):
        unit_vec, distances, mask = self._gather_neighbor_vectors(
            positions, angular_neighbors, angular_offsets
        )
        n_atoms = positions.shape[0]
        if unit_vec is None or not mask.any():
            return positions.new_zeros((n_atoms, self.n_desc, self.num_L))

        max_nn = angular_neighbors.shape[1]
        mask_flat = mask.reshape(-1)
        distances_flat = distances.reshape(-1)
        type_i = types.unsqueeze(1).expand(-1, max_nn).reshape(-1)
        safe_neighbors = angular_neighbors.clone()
        safe_neighbors[angular_neighbors == -1] = 0
        type_j = types[safe_neighbors].reshape(-1)

        f = chebyshev_basis(distances_flat, self.r_c, self.k_max)
        f = f * mask_flat.unsqueeze(-1)
        coeff = self.c_table[type_i, type_j]
        gn_flat = torch.sum(coeff * f.unsqueeze(1), dim=-1)
        gn = gn_flat.reshape(n_atoms, max_nn, self.n_desc)
        gn = gn * mask.unsqueeze(-1)

        basis_flat = self._compute_basis(unit_vec.reshape(-1, 3), mask_flat)
        basis = basis_flat.reshape(n_atoms, max_nn, NUM_OF_ABC)
        s = torch.einsum("ijn,ijb->inb", gn, basis)
        q = self._compute_q(s)
        return q


class Descriptor(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.radial = RadialDescriptor(
            n_types=para["n_types"],
            n_desc=para["n_desc_radial"],
            k_max=para["k_max_radial"],
            r_c=para["rcut_radial"],
        )
        l_max_4body = para.get("l_max_4body", 2)
        l_max_5body = para.get("l_max_5body", 0)
        self.angular = AngularDescriptor(
            n_types=para["n_types"],
            n_desc=para["n_desc_angular"],
            k_max=para["k_max_angular"],
            r_c=para["rcut_angular"],
            l_max=para["l_max"],
            l_max_4body=l_max_4body,
            l_max_5body=l_max_5body,
        )
        para["l_max_4body"] = l_max_4body
        para["l_max_5body"] = l_max_5body

    def forward(self, batch):
        g_radial = self.radial(
            batch["types"],
            batch["positions"],
            batch["radial_neighbors"],
            batch["radial_offsets"],
        )
        g_angular = self.angular(
            batch["types"],
            batch["positions"],
            batch["angular_neighbors"],
            batch["angular_offsets"],
        )
        return {"g_radial": g_radial, "g_angular": g_angular}
