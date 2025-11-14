import torch
import torch.nn as nn

NUM_OF_ABC = 80
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
  fc = torch.where(
    r < r_c,
    0.5 * torch.cos(torch.pi * r / r_c) + 0.5,
    torch.zeros_like(r),
  )
  x = 2.0 * (r / r_c - 1.0) ** 2 - 1.0
  half_fc = 0.5 * fc
  fn_list = [torch.ones_like(x)]
  if k_max > 1:
    fn_list.append(x)
    for _ in range(2, k_max):
      fn_list.append(2.0 * x * fn_list[-1] - fn_list[-2])
  fn = torch.stack(fn_list, dim=-1)
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

  def forward(self, types, positions, radial_neighbors):
    n_atoms, nn_radial = radial_neighbors.shape
    g = positions.new_zeros((n_atoms, self.n_desc))
    valid_mask = radial_neighbors != -1
    if not valid_mask.any():
      return g

    atom_indices = torch.arange(n_atoms, device=types.device).unsqueeze(1).expand(-1, nn_radial)
    valid_atom_indices = atom_indices[valid_mask]
    valid_neighbor_indices = radial_neighbors[valid_mask]

    pos_i = positions[valid_atom_indices]
    pos_j = positions[valid_neighbor_indices]
    distances = torch.linalg.norm(pos_j - pos_i, dim=-1)
    f = chebyshev_basis(distances, self.r_c, self.k_max)

    type_i = types[valid_atom_indices]
    type_j = types[valid_neighbor_indices]
    coeff = self.c_table[type_i, type_j]
    edge_descriptors = torch.sum(coeff * f.unsqueeze(1), dim=-1)

    g.index_add_(0, valid_atom_indices, edge_descriptors)
    return g


class AngularDescriptor(nn.Module):
  def __init__(self, n_types, n_desc, k_max, r_c, l_max, l_max_4body=2, l_max_5body=0):
    super().__init__()
    self.n_types = n_types
    self.n_desc = n_desc
    self.k_max = k_max
    self.r_c = r_c
    self.L_max = l_max
    self.L_max_4body = l_max_4body
    self.L_max_5body = l_max_5body
    self.include_4body = l_max_4body == 2
    self.include_5body = l_max_5body == 1
    self.num_L = self.L_max + int(self.include_4body) + int(self.include_5body)

    self.c_table = nn.Parameter(torch.randn(n_types, n_types, n_desc, k_max))
    self.register_buffer("C3B", torch.tensor(C3B_VALUES, dtype=torch.float32))
    self.register_buffer("C4B", torch.tensor(C4B_VALUES, dtype=torch.float32))
    self.register_buffer("C5B", torch.tensor(C5B_VALUES, dtype=torch.float32))
    for L, data in Z_COEFF_DATA.items():
      self.register_buffer(f"Z{L}", torch.tensor(data, dtype=torch.float32))

  @staticmethod
  def _complex_product(a, b, real, imag):
    real_temp = real
    real = a * real_temp - b * imag
    imag = a * imag + b * real_temp
    return real, imag

  def _accumulate_s_one(self, L, unit_vec, fn_value, s_row):
    x12, y12, z12 = unit_vec
    z_pow = [torch.ones_like(z12)]
    for _ in range(L):
      z_pow.append(z12 * z_pow[-1])

    s_index = L * L - 1
    real_part = x12
    imag_part = y12
    z_coeff = getattr(self, f"Z{L}")

    for n1 in range(L + 1):
      n2_start = 0 if (L + n1) % 2 == 0 else 1
      z_factor = torch.zeros_like(fn_value)
      for n2 in range(n2_start, L - n1 + 1, 2):
        z_factor = z_factor + z_coeff[n1, n2] * z_pow[n2]
      z_factor = z_factor * fn_value
      if n1 == 0:
        s_row[s_index] = s_row[s_index] + z_factor
        s_index += 1
      else:
        s_row[s_index] = s_row[s_index] + z_factor * real_part
        s_row[s_index + 1] = s_row[s_index + 1] + z_factor * imag_part
        s_index += 2
        real_part, imag_part = self._complex_product(x12, y12, real_part, imag_part)

  def _accumulate_s(self, r_vec, fn_value, s_row):
    d = torch.linalg.norm(r_vec)
    if float(d) <= 1.0e-8:
      return
    unit_vec = r_vec / d
    if self.L_max >= 1:
      self._accumulate_s_one(1, unit_vec, fn_value, s_row)
    if self.L_max >= 2:
      self._accumulate_s_one(2, unit_vec, fn_value, s_row)
    if self.L_max >= 3:
      self._accumulate_s_one(3, unit_vec, fn_value, s_row)
    if self.L_max >= 4:
      self._accumulate_s_one(4, unit_vec, fn_value, s_row)
    if self.L_max >= 5:
      self._accumulate_s_one(5, unit_vec, fn_value, s_row)
    if self.L_max >= 6:
      self._accumulate_s_one(6, unit_vec, fn_value, s_row)
    if self.L_max >= 7:
      self._accumulate_s_one(7, unit_vec, fn_value, s_row)
    if self.L_max >= 8:
      self._accumulate_s_one(8, unit_vec, fn_value, s_row)

  def _find_q_one(self, L, s_row):
    start = L * L - 1
    num_terms = 2 * L + 1
    terms = s_row[start : start + num_terms]
    weight = self.C3B[start : start + num_terms]
    q_val = torch.sum(weight[1:] * terms[1:] * terms[1:]) * 2.0
    q_val = q_val + weight[0] * terms[0] * terms[0]
    return q_val

  def _find_q(self, s_row):
    values = []
    for L in range(1, self.L_max + 1):
      values.append(self._find_q_one(L, s_row))
    if self.include_4body:
      s3, s4, s5, s6, s7 = s_row[3], s_row[4], s_row[5], s_row[6], s_row[7]
      q4 = (
        self.C4B[0] * s3 * s3 * s3
        + self.C4B[1] * s3 * (s4 * s4 + s5 * s5)
        + self.C4B[2] * s3 * (s6 * s6 + s7 * s7)
        + self.C4B[3] * s6 * (s5 * s5 - s4 * s4)
        + self.C4B[4] * s4 * s5 * s7
      )
      values.append(q4)
    if self.include_5body:
      s0_sq = s_row[0] * s_row[0]
      s1_sq_plus_s2_sq = s_row[1] * s_row[1] + s_row[2] * s_row[2]
      q5 = (
        self.C5B[0] * s0_sq * s0_sq
        + self.C5B[1] * s0_sq * s1_sq_plus_s2_sq
        + self.C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq
      )
      values.append(q5)
    return torch.stack(values) if values else s_row.new_zeros(0)

  def forward(self, types, positions, angular_neighbors):
    n_atoms, _ = angular_neighbors.shape
    result = positions.new_zeros((n_atoms, self.n_desc, self.num_L))

    for i in range(n_atoms):
      neighbor_indices = angular_neighbors[i]
      neighbor_indices = neighbor_indices[neighbor_indices >= 0]
      if neighbor_indices.numel() == 0:
        continue

      s_all = positions.new_zeros((self.n_desc, NUM_OF_ABC))
      type_i = int(types[i].item())
      neighbor_ids = neighbor_indices.to(torch.long)
      r_vecs = positions[neighbor_ids] - positions[i]
      neighbor_types = types[neighbor_ids]

      for r_vec, type_j in zip(r_vecs, neighbor_types):
        dist = torch.linalg.norm(r_vec)
        fn = chebyshev_basis(dist.view(1), self.r_c, self.k_max)[0]
        coeff = self.c_table[type_i, int(type_j.item())]
        gn_all = torch.matmul(coeff, fn)
        for n in range(self.n_desc):
          self._accumulate_s(r_vec, gn_all[n], s_all[n])

      for n in range(self.n_desc):
        result[i, n] = self._find_q(s_all[n])

    return result


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
    )
    g_angular = self.angular(
      batch["types"],
      batch["positions"],
      batch["angular_neighbors"],
    )
    return {"g_radial": g_radial, "g_angular": g_angular}