import numpy as np
from itertools import product


def _cell_thickness(cell):
    a, b, c = cell
    volume = abs(np.dot(a, np.cross(b, c)))
    area_a = np.linalg.norm(np.cross(b, c))
    area_b = np.linalg.norm(np.cross(c, a))
    area_c = np.linalg.norm(np.cross(a, b))
    if area_a < 1.0e-12 or area_b < 1.0e-12 or area_c < 1.0e-12:
        raise ValueError("Degenerate cell is not supported for periodic neighbor construction.")
    return volume / np.array([area_a, area_b, area_c], dtype=np.float64)


def _pack_neighbor_data(neighbor_ids, neighbor_offsets, n_atoms, max_neighbors):
    nbr = np.full((n_atoms, max_neighbors), -1, dtype=np.int64)
    offsets = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)
    for i in range(n_atoms):
        count = len(neighbor_ids[i])
        if count > max_neighbors:
            raise ValueError(
                f"Neighbor count {count} exceeds max_neighbors={max_neighbors}. "
                "Increase NN_radial/NN_angular."
            )
        if count == 0:
            continue
        nbr[i, :count] = np.asarray(neighbor_ids[i], dtype=np.int64)
        offsets[i, :count] = np.asarray(neighbor_offsets[i], dtype=np.float32)
    return nbr, offsets


def build_neighbor_list(
    positions,
    cell,
    pbc,
    cutoff_radial,
    cutoff_angular,
    max_neighbors_radial,
    max_neighbors_angular,
):
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    pbc = np.asarray(pbc, dtype=bool)
    n_atoms = positions.shape[0]

    if positions.shape != (n_atoms, 3):
        raise ValueError(f"positions should have shape (N, 3), got {positions.shape}.")
    if cell.shape != (3, 3):
        raise ValueError(f"cell should have shape (3, 3), got {cell.shape}.")

    if n_atoms == 0:
        radial_neighbors = np.full((0, max_neighbors_radial), -1, dtype=np.int64)
        radial_offsets = np.zeros((0, max_neighbors_radial, 3), dtype=np.float32)
        angular_neighbors = np.full((0, max_neighbors_angular), -1, dtype=np.int64)
        angular_offsets = np.zeros((0, max_neighbors_angular, 3), dtype=np.float32)
        return radial_neighbors, radial_offsets, angular_neighbors, angular_offsets

    rc_max = max(float(cutoff_radial), float(cutoff_angular))
    thickness = _cell_thickness(cell)
    num_cells = np.ones(3, dtype=np.int64)
    for d in range(3):
        if pbc[d]:
            num_cells[d] = max(1, int(np.ceil(2.0 * rc_max / thickness[d])))

    expanded_cell = cell.copy()
    expanded_cell[0] *= num_cells[0]
    expanded_cell[1] *= num_cells[1]
    expanded_cell[2] *= num_cells[2]
    inv_expanded_cell = np.linalg.inv(expanded_cell)

    base_delta = positions[None, :, :] - positions[:, None, :]
    eye_mask = np.eye(n_atoms, dtype=bool)
    cutoff_radial_sq = float(cutoff_radial) ** 2
    cutoff_angular_sq = float(cutoff_angular) ** 2

    radial_ids = [[] for _ in range(n_atoms)]
    radial_offsets = [[] for _ in range(n_atoms)]
    angular_ids = [[] for _ in range(n_atoms)]
    angular_offsets = [[] for _ in range(n_atoms)]

    range_a = range(num_cells[0]) if pbc[0] else range(1)
    range_b = range(num_cells[1]) if pbc[1] else range(1)
    range_c = range(num_cells[2]) if pbc[2] else range(1)

    for ia, ib, ic in product(range_a, range_b, range_c):
        shift = ia * cell[0] + ib * cell[1] + ic * cell[2]
        delta = base_delta + shift
        frac = delta @ inv_expanded_cell
        if pbc[0]:
            frac[..., 0] -= np.rint(frac[..., 0])
        if pbc[1]:
            frac[..., 1] -= np.rint(frac[..., 1])
        if pbc[2]:
            frac[..., 2] -= np.rint(frac[..., 2])
        delta_mic = frac @ expanded_cell
        dist2 = np.einsum("ijk,ijk->ij", delta_mic, delta_mic)

        mask_radial = dist2 < cutoff_radial_sq
        mask_angular = dist2 < cutoff_angular_sq
        if ia == 0 and ib == 0 and ic == 0:
            mask_radial &= ~eye_mask
            mask_angular &= ~eye_mask

        for i in range(n_atoms):
            cols_radial = np.flatnonzero(mask_radial[i])
            if cols_radial.size:
                radial_ids[i].extend(cols_radial.tolist())
                offsets = delta_mic[i, cols_radial] - base_delta[i, cols_radial]
                radial_offsets[i].extend(offsets.astype(np.float32))

            cols_angular = np.flatnonzero(mask_angular[i])
            if cols_angular.size:
                angular_ids[i].extend(cols_angular.tolist())
                offsets = delta_mic[i, cols_angular] - base_delta[i, cols_angular]
                angular_offsets[i].extend(offsets.astype(np.float32))

    radial_neighbors, radial_offsets = _pack_neighbor_data(
        radial_ids, radial_offsets, n_atoms, max_neighbors_radial
    )
    angular_neighbors, angular_offsets = _pack_neighbor_data(
        angular_ids, angular_offsets, n_atoms, max_neighbors_angular
    )
    return radial_neighbors, radial_offsets, angular_neighbors, angular_offsets
