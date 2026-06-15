from ase import Atoms
from ase.geometry import get_distances
import numpy as np

__all__ = ["wigner_seitz", "unwrap", "find_asd", "find_msd"]


def wigner_seitz(atoms: Atoms, reference_atoms: Atoms, chunk_size: int = 256) -> dict:
    """
    Assign atoms to the nearest reference lattice sites using Wigner-Seitz cells.

    The reference atoms define the lattice sites. A site with no assigned atom is
    a vacancy; a site with more than one assigned atom contains interstitial atoms.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError('atoms must be an ASE Atoms object.')
    if not isinstance(reference_atoms, Atoms):
        raise TypeError('reference_atoms must be an ASE Atoms object.')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer.')

    cell = reference_atoms.get_cell()
    pbc = reference_atoms.get_pbc()
    site_positions = reference_atoms.get_positions()

    scaled_positions = atoms.get_scaled_positions(wrap=True)
    atom_positions = np.dot(scaled_positions, cell)

    atom_sites = np.empty(len(atoms), dtype=int)
    atom_distances = np.empty(len(atoms), dtype=float)

    for start in range(0, len(atoms), chunk_size):
        stop = min(start + chunk_size, len(atoms))
        _, distances = get_distances(
            atom_positions[start:stop],
            site_positions,
            cell=cell,
            pbc=pbc,
        )
        nearest_sites = np.argmin(distances, axis=1)
        atom_sites[start:stop] = nearest_sites
        atom_distances[start:stop] = distances[np.arange(stop - start), nearest_sites]

    site_atoms = [[] for _ in range(len(reference_atoms))]
    for atom_index, site_index in enumerate(atom_sites):
        site_atoms[site_index].append(atom_index)

    occupants = np.full(len(reference_atoms), -1, dtype=int)
    vacancies = []
    interstitials = []
    interstitial_sites = []

    for site_index, atom_indices in enumerate(site_atoms):
        if not atom_indices:
            vacancies.append(site_index)
            continue

        atom_indices = sorted(atom_indices, key=lambda index: atom_distances[index])
        occupants[site_index] = atom_indices[0]
        for atom_index in atom_indices[1:]:
            interstitials.append(atom_index)
            interstitial_sites.append(site_index)

    return {
        'vacancies': np.array(vacancies, dtype=int),
        'interstitials': np.array(interstitials, dtype=int),
        'interstitial_sites': np.array(interstitial_sites, dtype=int),
        'occupants': occupants,
        'site_atoms': site_atoms,
        'atom_sites': atom_sites,
        'atom_distances': atom_distances,
    }


def unwrap(frames: list) -> list:
    """Unwrap a trajectory by removing jumps across periodic boundaries."""
    diffs = []
    for i in range(1, len(frames)):
        cell = frames[i].get_cell().diagonal()
        diff = frames[i].positions - frames[i - 1].positions
        diff -= np.round(diff / cell) * cell
        diffs.append(diff)
    for i in range(1, len(frames)):
        frames[i].positions = frames[i - 1].positions + diffs[i - 1]
    return frames


def find_asd(frames: list, *symbols):
    """Average squared displacement from the first frame."""
    if not symbols:
        asd = []
        num_atoms = len(frames[0])
        for frame in frames:
            displacement = frames[0].positions - frame.positions
            asd.append(np.sum(displacement ** 2) / num_atoms)
        return asd

    asd = {}
    for symbol in symbols:
        indices = [atom.index for atom in frames[0] if atom.symbol == symbol]
        num_atoms = len(indices)
        asd[symbol] = []
        for frame in frames:
            displacement = frames[0].positions[indices] - frame.positions[indices]
            asd[symbol].append(np.sum(displacement ** 2) / num_atoms)

    asd['average'] = []
    for frame in frames:
        displacement = frames[0].positions - frame.positions
        asd['average'].append(np.sum(displacement ** 2) / len(frames[0]))
    return asd


def find_msd(frames: list, Nc: int = 100, *symbols):
    """Mean squared displacement averaged over time origins."""
    Nd = len(frames)
    Nc = min(Nc, Nd)
    if not symbols:
        msd = []
        num_atoms = len(frames[0])
        for n in range(Nc):
            value = 0
            for m in range(Nd - n):
                value += np.sum((frames[m].positions - frames[m + n].positions) ** 2)
            msd.append(value / ((Nd - n) * num_atoms))
        return msd

    msd = {}
    for symbol in symbols:
        indices = [atom.index for atom in frames[0] if atom.symbol == symbol]
        num_atoms = len(indices)
        msd[symbol] = []
        for n in range(Nc):
            value = 0
            for m in range(Nd - n):
                value += np.sum((frames[m].positions[indices] - frames[m + n].positions[indices]) ** 2)
            msd[symbol].append(value / ((Nd - n) * num_atoms))

    msd['average'] = []
    for n in range(Nc):
        value = 0
        for m in range(Nd - n):
            value += np.sum((frames[m].positions - frames[m + n].positions) ** 2)
        msd['average'].append(value / ((Nd - n) * len(frames[0])))
    return msd
