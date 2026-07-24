from dataclasses import dataclass
from math import comb

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
C4B2_VALUES = [
    0.027493550848847,
    0.164961305093080,
    -0.013746775424423,
    0.041240326273270,
    0.082480652546540,
]
C4B_123_VALUES = [
    -0.008418146349617,
    -0.016836292699234,
    -0.033672585398469,
    -0.042090731748086,
    -0.067345170796937,
    -0.084181463496172,
    -0.168362926992344,
]
C4B_233_VALUES = [
    0.008572620635186,
    0.009644198214584,
    0.019288396429168,
    0.025717861905558,
    0.026789439484956,
    0.032147327381947,
    0.038576792858337,
    0.128589309527790,
    0.192883964291685,
    0.321473273819474,
]
C4B_134_VALUES = [
    0.003645164295772,
    0.004860219061029,
    0.006075273826286,
    0.018225821478859,
    0.024301095305146,
    0.036451642957719,
    0.042526916784005,
    0.072903285915437,
    0.085053833568010,
    0.255161500704030,
]

_Q123_GROUPS = [
    (6, [(1, (12, 2, 4)), (-1, (11, 2, 5)), (1, (1, 11, 4)), (1, (1, 12, 5))]),
    (5, [(1, (0, 11, 6)), (1, (0, 12, 7))]),
    (3, [(1, (14, 2, 6)), (-1, (13, 2, 7)), (1, (1, 13, 6)), (1, (1, 14, 7))]),
    (4, [(1, (10, 0, 5)), (1, (0, 4, 9))]),
    (1, [(1, (10, 2, 3)), (1, (0, 3, 8)), (1, (1, 3, 9))]),
    (0, [(1, (10, 2, 6)), (-1, (10, 1, 7)), (-1, (2, 7, 9)), (-1, (1, 6, 9))]),
    (2, [(-1, (2, 5, 8)), (-1, (1, 4, 8))]),
]
_Q233_GROUPS = [
    (0, [(1, (3, 8, 8))]),
    (1, [(1, (10, 10, 3)), (1, (3, 9, 9))]),
    (2, [(-1, (10, 10, 6)), (1, (6, 9, 9))]),
    (3, [(1, (4, 8, 9)), (1, (10, 5, 8))]),
    (4, [(-1, (13, 13, 3)), (-1, (14, 14, 3))]),
    (5, [(-1, (14, 7, 9)), (-1, (13, 6, 9)), (-1, (10, 14, 6)), (1, (10, 13, 7))]),
    (6, [(1, (10, 7, 9))]),
    (7, [(-1, (11, 6, 8)), (-1, (12, 7, 8))]),
    (8, [(1, (11, 4, 9)), (1, (12, 5, 9)), (1, (10, 12, 4)), (-1, (10, 11, 5))]),
    (9, [(1, (12, 14, 4)), (1, (11, 14, 5)), (1, (13, 11, 4)), (-1, (13, 12, 5))]),
]
_Q134_GROUPS = [
    (0, [(-1, (10, 15, 2)), (-1, (1, 15, 9))]),
    (1, [(1, (0, 15, 8))]),
    (2, [(-1, (1, 13, 18)), (-1, (1, 14, 19)), (-1, (2, 14, 18)), (1, (2, 13, 19))]),
    (3, [(-1, (10, 18, 2)), (1, (1, 10, 19)), (1, (1, 18, 9)), (1, (2, 19, 9))]),
    (4, [(1, (1, 16, 8)), (1, (2, 17, 8))]),
    (5, [(1, (0, 10, 17)), (1, (0, 16, 9)), (-1, (1, 11, 16)), (-1, (1, 12, 17)),
         (-1, (2, 12, 16)), (1, (2, 11, 17))]),
    (6, [(1, (1, 13, 22)), (1, (1, 14, 23)), (-1, (2, 14, 22)), (1, (2, 13, 23))]),
    (7, [(1, (0, 11, 18)), (1, (0, 12, 19))]),
    (8, [(1, (0, 13, 20)), (1, (0, 14, 21))]),
    (9, [(1, (1, 11, 20)), (1, (1, 12, 21)), (-1, (2, 12, 20)), (1, (2, 11, 21))]),
]


def _expand_invariant_terms(coefficients, groups):
    return [
        (sign * coefficients[coefficient_index], indices)
        for coefficient_index, group in groups
        for sign, indices in group
    ]


Q123_TERMS = _expand_invariant_terms(C4B_123_VALUES, _Q123_GROUPS)
Q233_TERMS = _expand_invariant_terms(C4B_233_VALUES, _Q233_GROUPS)
Q134_TERMS = _expand_invariant_terms(C4B_134_VALUES, _Q134_GROUPS)
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


def _invariant_gradient_data(terms):
    coefficients = []
    output_indices = []
    left_indices = []
    right_indices = []
    for coefficient, indices in terms:
        if len(indices) != 3:
            raise ValueError("Extended angular invariants must be cubic.")
        for position, output_index in enumerate(indices):
            remaining = [
                index
                for other_position, index in enumerate(indices)
                if other_position != position
            ]
            coefficients.append(coefficient)
            output_indices.append(output_index)
            left_indices.append(remaining[0])
            right_indices.append(remaining[1])
    return (
        torch.tensor(coefficients, dtype=torch.float64),
        torch.tensor(output_indices, dtype=torch.long),
        torch.tensor(left_indices, dtype=torch.long),
        torch.tensor(right_indices, dtype=torch.long),
    )


def _complex_power_terms(power, imaginary):
    terms = []
    for y_power in range(power + 1):
        if imaginary:
            if y_power % 2 == 0:
                continue
            sign = -1.0 if ((y_power - 1) // 2) % 2 else 1.0
        else:
            if y_power % 2:
                continue
            sign = -1.0 if (y_power // 2) % 2 else 1.0
        terms.append(
            (
                sign * comb(power, y_power),
                power - y_power,
                y_power,
            )
        )
    return terms


def _angular_basis_derivative_data(maximum):
    components = []
    for order in range(1, maximum + 1):
        z_coefficients = Z_COEFF_DATA[order]
        for xy_power in range(order + 1):
            z_terms = [
                (coefficient, z_power)
                for z_power, coefficient in enumerate(z_coefficients[xy_power])
                if coefficient != 0.0
            ]
            variants = [False] if xy_power == 0 else [False, True]
            for imaginary in variants:
                terms = {}
                for xy_coefficient, x_power, y_power in _complex_power_terms(
                    xy_power, imaginary
                ):
                    for z_coefficient, z_power in z_terms:
                        exponent = (x_power, y_power, z_power)
                        terms[exponent] = (
                            terms.get(exponent, 0.0)
                            + xy_coefficient * z_coefficient
                        )
                components.append(terms)

    derivative_components = [[], [], []]
    exponent_set = set()
    for component_index, terms in enumerate(components):
        for exponent, coefficient in terms.items():
            for axis in range(3):
                power = exponent[axis]
                if power == 0:
                    continue
                derivative_exponent = list(exponent)
                derivative_exponent[axis] -= 1
                derivative_exponent = tuple(derivative_exponent)
                exponent_set.add(derivative_exponent)
                derivative_components[axis].append(
                    (
                        derivative_exponent,
                        component_index,
                        power * coefficient,
                    )
                )

    exponents = sorted(exponent_set, key=lambda value: (sum(value), value))
    exponent_lookup = {
        exponent: index for index, exponent in enumerate(exponents)
    }
    coefficients = torch.zeros(
        (3, len(exponents), len(components)), dtype=torch.float32
    )
    for axis, entries in enumerate(derivative_components):
        for exponent, component_index, coefficient in entries:
            coefficients[
                axis, exponent_lookup[exponent], component_index
            ] += coefficient
    return torch.tensor(exponents, dtype=torch.long), coefficients


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


def chebyshev_basis_and_derivative(r, r_c, k_max):
    """Evaluate the GPUMD Chebyshev basis and ``d f_k / d r``."""
    rcinv = 1.0 / r_c
    arg = torch.pi * r * rcinv
    fc = 0.5 * torch.cos(arg) + 0.5
    fcp = -0.5 * torch.pi * rcinv * torch.sin(arg)
    r_shift = r * rcinv - 1.0
    x = 2.0 * r_shift * r_shift - 1.0
    dxdr = 4.0 * r_shift * rcinv

    basis = torch.empty((*r.shape, k_max), dtype=r.dtype, device=r.device)
    derivative = torch.empty_like(basis)
    if k_max == 0:
        return basis, derivative

    basis[..., 0] = fc
    derivative[..., 0] = fcp
    if k_max == 1:
        return basis, derivative

    t_prev2 = torch.ones_like(r)
    t_prev1 = x
    u_prev2 = torch.ones_like(r)
    u_prev1 = 2.0 * x
    core = 0.5 * (x + 1.0)
    basis[..., 1] = core * fc
    derivative[..., 1] = 0.5 * dxdr * fc + core * fcp

    for k in range(2, k_max):
        t_next = 2.0 * x * t_prev1 - t_prev2
        u_next = 2.0 * x * u_prev1 - u_prev2
        core = 0.5 * (t_next + 1.0)
        basis[..., k] = core * fc
        derivative[..., k] = 0.5 * k * u_prev1 * dxdr * fc + core * fcp
        t_prev2, t_prev1 = t_prev1, t_next
        u_prev2, u_prev1 = u_prev1, u_next

    return basis, derivative


@dataclass
class PairState:
    center: torch.Tensor
    neighbor: torch.Tensor
    vectors: torch.Tensor
    inverse_distance: torch.Tensor
    basis_derivative: torch.Tensor


@dataclass
class AngularState(PairState):
    basis: torch.Tensor
    radial_values: torch.Tensor
    moments: torch.Tensor


@dataclass
class DescriptorState:
    types: torch.Tensor
    values: torch.Tensor
    radial_values: torch.Tensor
    angular_values: torch.Tensor
    radial: PairState
    angular: AngularState


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
    def __init__(
        self,
        n_types,
        n_desc,
        k_max,
        r_c,
        l_max,
        has_q_222=1,
        has_q_1111=0,
        has_q_112=0,
        has_q_123=0,
        has_q_233=0,
        has_q_134=0,
    ):
        super().__init__()
        self.n_types = n_types
        self.n_desc = n_desc
        self.k_max = k_max
        self.r_c = r_c
        self.L_max = l_max
        self.has_q_222 = int(has_q_222 > 0)
        self.has_q_1111 = int(has_q_1111 > 0)
        self.has_q_112 = int(has_q_112 > 0)
        self.has_q_123 = int(has_q_123 > 0)
        self.has_q_233 = int(has_q_233 > 0)
        self.has_q_134 = int(has_q_134 > 0)
        if (self.has_q_123 or self.has_q_233) and self.L_max < 3:
            raise ValueError("q_123 and q_233 require l_max >= 3.")
        if self.has_q_134 and self.L_max < 4:
            raise ValueError("q_134 requires l_max >= 4.")
        self.num_L = self.L_max + sum(
            (
                self.has_q_222,
                self.has_q_1111,
                self.has_q_112,
                self.has_q_123,
                self.has_q_233,
                self.has_q_134,
            )
        )

        self.c_table = nn.Parameter(torch.randn(n_types, n_types, n_desc, k_max))
        self.register_buffer("C3B", torch.tensor(C3B_VALUES, dtype=torch.float32))
        self.register_buffer("C4B", torch.tensor(C4B_VALUES, dtype=torch.float32))
        self.register_buffer("C5B", torch.tensor(C5B_VALUES, dtype=torch.float32))
        self.register_buffer("C4B2", torch.tensor(C4B2_VALUES, dtype=torch.float32))
        for L, data in Z_COEFF_DATA.items():
            tensor = torch.tensor(data, dtype=torch.float32)
            self.register_buffer(f"Z{L}", tensor)
        derivative_exponents, derivative_coefficients = (
            _angular_basis_derivative_data(self.L_max)
        )
        self.register_buffer(
            "basis_derivative_exponents",
            derivative_exponents,
            persistent=False,
        )
        self.register_buffer(
            "basis_derivative_coefficients",
            derivative_coefficients,
            persistent=False,
        )
        for name, terms in (
            ("q123", Q123_TERMS),
            ("q233", Q233_TERMS),
            ("q134", Q134_TERMS),
        ):
            gradient_data = _invariant_gradient_data(terms)
            for suffix, tensor in zip(
                ("coefficients", "output", "left", "right"),
                gradient_data,
            ):
                self.register_buffer(
                    f"{name}_gradient_{suffix}",
                    tensor,
                    persistent=False,
                )

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

        if self.has_q_222:
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

        if self.has_q_1111:
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

        if self.has_q_112:
            if self.L_max < 2:
                raise ValueError("L_max must be >= 2 to include q_112.")
            s10, s11r, s11i = s[..., 0], s[..., 1], s[..., 2]
            s20, s21r, s21i = s[..., 3], s[..., 4], s[..., 5]
            s22r, s22i = s[..., 6], s[..., 7]
            cb = self.C4B2
            q112 = (
                cb[0] * s10 * s10 * s20
                + cb[1] * s10 * (s11r * s21r + s11i * s21i)
                + cb[2] * s20 * (s11r * s11r + s11i * s11i)
                + cb[3] * s22r * (s11r * s11r - s11i * s11i)
                + cb[4] * s11r * s11i * s22i
            )
            q_list.append(q112)

        if self.has_q_123:
            q_list.append(self._evaluate_invariant(s, Q123_TERMS))
        if self.has_q_233:
            q_list.append(self._evaluate_invariant(s, Q233_TERMS))
        if self.has_q_134:
            q_list.append(self._evaluate_invariant(s, Q134_TERMS))

        return torch.stack(q_list, dim=-1) if q_list else s.new_zeros((s.shape[0], s.shape[1], 0))

    @staticmethod
    def _evaluate_invariant(s, terms):
        result = torch.zeros_like(s[..., 0])
        for coefficient, indices in terms:
            term = s[..., indices[0]]
            for index in indices[1:]:
                term = term * s[..., index]
            result = result + coefficient * term
        return result

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
        has_q_222 = int(para.get("has_q_222", para.get("l_max_4body", 2) > 0))
        has_q_1111 = int(para.get("has_q_1111", para.get("l_max_5body", 0) > 0))
        has_q_112 = int(para.get("has_q_112", 0))
        has_q_123 = int(para.get("has_q_123", 0))
        has_q_233 = int(para.get("has_q_233", 0))
        has_q_134 = int(para.get("has_q_134", 0))
        self.angular = AngularDescriptor(
            n_types=para["n_types"],
            n_desc=para["n_desc_angular"],
            k_max=para["k_max_angular"],
            r_c=para["rcut_angular"],
            l_max=para["l_max"],
            has_q_222=has_q_222,
            has_q_1111=has_q_1111,
            has_q_112=has_q_112,
            has_q_123=has_q_123,
            has_q_233=has_q_233,
            has_q_134=has_q_134,
        )
        para["l_max_4body"] = 2 if has_q_222 else 0
        para["l_max_5body"] = has_q_1111
        para["has_q_222"] = has_q_222
        para["has_q_1111"] = has_q_1111
        para["has_q_112"] = has_q_112
        para["has_q_123"] = has_q_123
        para["has_q_233"] = has_q_233
        para["has_q_134"] = has_q_134

    @staticmethod
    def _pair_geometry(types, positions, neighbors, offsets):
        mask = neighbors != -1
        max_neighbors = neighbors.shape[1]
        center_grid = torch.arange(
            positions.shape[0], device=positions.device
        ).unsqueeze(1).expand(-1, max_neighbors)
        center = center_grid[mask]
        neighbor = neighbors[mask]
        vectors = (
            positions[neighbor]
            + offsets.to(device=positions.device, dtype=positions.dtype)[mask]
            - positions[center]
        )
        distances = torch.linalg.norm(vectors, dim=-1)
        inverse_distance = torch.reciprocal(torch.clamp(distances, min=1.0e-12))
        return center, neighbor, vectors, distances, inverse_distance

    def compute_state(self, batch, track_geometry=False):
        types = batch["types"]
        positions = batch["positions"]
        radial_offsets = batch["radial_offsets"]
        angular_offsets = batch["angular_offsets"]
        if not track_geometry:
            positions = positions.detach()
            radial_offsets = radial_offsets.detach()
            angular_offsets = angular_offsets.detach()

        n_atoms = positions.shape[0]
        pi_rad, pj_rad, rij_rad, dij_rad, inv_rad = self._pair_geometry(
            types,
            positions,
            batch["radial_neighbors"],
            radial_offsets,
        )
        fk_rad, fkp_rad = chebyshev_basis_and_derivative(
            dij_rad, self.radial.r_c, self.radial.k_max
        )
        if pi_rad.numel():
            c_rad = self.radial.c_table[types[pi_rad], types[pj_rad]]
            pair_radial = torch.einsum("pnk,pk->pn", c_rad, fk_rad)
            q_radial = positions.new_zeros((n_atoms, self.radial.n_desc))
            q_radial.scatter_add_(
                0, pi_rad.unsqueeze(-1).expand_as(pair_radial), pair_radial
            )
        else:
            q_radial = positions.new_zeros((n_atoms, self.radial.n_desc))
            q_radial = q_radial + self.radial.c_table.sum() * 0.0

        pi_ang, pj_ang, rij_ang, dij_ang, inv_ang = self._pair_geometry(
            types,
            positions,
            batch["angular_neighbors"],
            angular_offsets,
        )
        fk_ang, fkp_ang = chebyshev_basis_and_derivative(
            dij_ang, self.angular.r_c, self.angular.k_max
        )
        num_lm = self.angular.L_max * (self.angular.L_max + 2)
        if pi_ang.numel():
            c_ang = self.angular.c_table[types[pi_ang], types[pj_ang]]
            gn_ang = torch.einsum("pnk,pk->pn", c_ang, fk_ang)
            unit = rij_ang * inv_ang.unsqueeze(-1)
            blm = self.angular._compute_basis(unit, None)[:, :num_lm]
            pair_moments = gn_ang.unsqueeze(-1) * blm.unsqueeze(1)
            moments = positions.new_zeros(
                (n_atoms, self.angular.n_desc, num_lm)
            )
            moments.scatter_add_(
                0,
                pi_ang.unsqueeze(-1).unsqueeze(-1).expand_as(pair_moments),
                pair_moments,
            )
        else:
            gn_ang = positions.new_zeros((0, self.angular.n_desc))
            blm = positions.new_zeros((0, num_lm))
            moments = positions.new_zeros(
                (n_atoms, self.angular.n_desc, num_lm)
            )
            moments = moments + self.angular.c_table.sum() * 0.0

        q_angular = self.angular._compute_q(moments)
        values = torch.cat(
            [
                q_radial,
                q_angular.permute(0, 2, 1).reshape(n_atoms, -1),
            ],
            dim=-1,
        )
        return DescriptorState(
            types=types,
            values=values,
            radial_values=q_radial,
            angular_values=q_angular,
            radial=PairState(
                center=pi_rad,
                neighbor=pj_rad,
                vectors=rij_rad,
                inverse_distance=inv_rad,
                basis_derivative=fkp_rad,
            ),
            angular=AngularState(
                center=pi_ang,
                neighbor=pj_ang,
                vectors=rij_ang,
                inverse_distance=inv_ang,
                basis_derivative=fkp_ang,
                basis=blm,
                radial_values=gn_ang,
                moments=moments,
            ),
        )

    def contract_forces(self, energy_gradient, state, compute_virial=True):
        from .force import contract_forces

        return contract_forces(
            self,
            energy_gradient,
            state,
            compute_virial=compute_virial,
        )

    def forward(self, batch):
        state = self.compute_state(batch, track_geometry=True)
        return {
            "g_radial": state.radial_values,
            "g_angular": state.angular_values,
        }
