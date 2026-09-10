"""Crystal geometry and known lattice-defect configurations."""

from collections import Counter

import numpy as np
import pytest
from ase import Atom, Atoms
from numpy.testing import assert_allclose, assert_array_equal

from wizard.structure.analysis import wigner_seitz
from wizard.structure.atoms import AlloyInfo, Morph


@pytest.mark.parametrize(
    "symbol,lattice,constants,count,volume",
    [
        ("Fe", "bcc", (3.0,), 16, 216.0),
        ("Cu", "fcc", (3.6,), 32, 8 * 3.6**3),
        ("Ti", "hcp", (2.9, 4.6), 16, 8 * np.sqrt(3) / 2 * 2.9**2 * 4.6),
    ],
)
def test_bulk_crystal_geometry(symbol, lattice, constants, count, volume):
    atoms = AlloyInfo(symbol, lattice, *constants).create_bulk_atoms((2, 2, 2))

    assert isinstance(atoms, Atoms)
    assert len(atoms) == count
    assert set(atoms.get_chemical_symbols()) == {symbol}
    assert atoms.get_volume() == pytest.approx(volume)
    assert_array_equal(atoms.pbc, [True, True, True])
    assert atoms.info["config_type"] == f"{lattice}_bulk"
    scaled = atoms.get_scaled_positions()
    assert np.all((scaled >= 0.0) & (scaled < 1.0))


def test_exactly_representable_alloy_composition():
    atoms = AlloyInfo("FeNi", "bcc", 3.0).create_bulk_atoms((2, 2, 2))
    reference = AlloyInfo("Fe", "bcc", 3.0).create_bulk_atoms((2, 2, 2))

    assert Counter(atoms.get_chemical_symbols()) == {"Fe": 8, "Ni": 8}
    assert_allclose(atoms.positions, reference.positions)
    assert_allclose(atoms.cell, reference.cell)


def test_interstitial_addition_preserves_host_and_counts():
    alloy = AlloyInfo("Fe", "bcc", 3.0)
    host = alloy.create_bulk_atoms((2, 2, 2))
    atoms = alloy.create_interstitial_atoms(
        (2, 2, 2),
        interstitials=[
            {"symbol": "C", "type": "oct", "num": 2},
            {"symbol": "H", "type": "tet", "num": 3},
        ],
    )

    assert Counter(atoms.get_chemical_symbols()) == {"Fe": 16, "C": 2, "H": 3}
    assert_allclose(atoms.positions[: len(host)], host.positions)
    assert_allclose(atoms.cell, host.cell)
    assert_array_equal(atoms.pbc, host.pbc)
    distances = atoms.get_all_distances(mic=True)
    assert np.min(distances[np.triu_indices(len(atoms), k=1)]) > 0.1


@pytest.mark.parametrize("add_interstitial", [False, True])
def test_wigner_seitz_known_vacancy_and_interstitial(add_interstitial):
    reference = AlloyInfo("Fe", "bcc", 3.0).create_bulk_atoms((2, 2, 2))
    atoms = reference.copy()
    Morph(atoms).create_vacancy(index=0)
    if add_interstitial:
        atoms.append(Atom("Fe", position=reference.positions[1] + [0.1, 0.0, 0.0]))

    defects = wigner_seitz(atoms, reference, chunk_size=3)

    assert_array_equal(defects["vacancies"], [0])
    assert len(defects["interstitials"]) == int(add_interstitial)
    if add_interstitial:
        assert_array_equal(defects["interstitials"], [len(atoms) - 1])
        assert_array_equal(defects["interstitial_sites"], [1])
    assert len(defects["interstitials"]) - len(defects["vacancies"]) == len(atoms) - len(reference)
    assert len(reference) == 16


def test_wigner_seitz_affine_strain_and_periodic_image():
    reference = AlloyInfo("Fe", "bcc", 3.0).create_bulk_atoms((2, 2, 2))
    atoms = reference.copy()
    atoms.set_cell([[6.3, 0.2, 0.0], [0.0, 5.9, 0.1], [0.0, 0.0, 6.1]], scale_atoms=True)
    atoms.positions[0] += atoms.cell[0]

    defects = wigner_seitz(atoms, reference, chunk_size=3)

    assert len(defects["vacancies"]) == 0
    assert len(defects["interstitials"]) == 0
    assert_array_equal(defects["atom_sites"], np.arange(16))
    assert_allclose(defects["atom_distances"], 0.0, atol=1e-12)
