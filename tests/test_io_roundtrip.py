"""Public extended-XYZ data exchange, including an independent ASE reader."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from numpy.testing import assert_allclose, assert_array_equal

from wizard.utils.io import dump_xyz, read_xyz


@pytest.mark.parametrize("mag", [[2.2, -0.4], [[0.0, 0.0, 2.2], [0.1, -0.2, -0.4]]])
def test_xyz_append_and_ase_interoperability(tmp_path, mag):
    atoms = Atoms(
        "FeNi",
        positions=[[0.1, 0.2, 0.3], [1.5, 2.0, 2.5]],
        cell=[[3.0, 0.0, 0.0], [0.25, 4.0, 0.0], [0.1, 0.2, 5.0]],
        pbc=[True, False, True],
    )
    forces = np.array([[0.2, -0.3, 0.4], [-0.2, 0.3, -0.4]])
    stress = np.array([[0.01, 0.004, 0.005], [0.004, 0.02, 0.006], [0.005, 0.006, 0.03]])
    groups = np.array([[0, 1], [2, 2]])
    atoms.set_velocities([[0.01, 0.02, 0.03], [-0.01, -0.02, -0.03]])
    atoms.info.update(
        energy=-8.25,
        forces=forces,
        stress=np.array([0.01, 0.02, 0.03, 0.006, 0.005, 0.004]),
        group=groups,
        mag=np.asarray(mag),
        config_type="alloy_bulk",
        weight=1.25,
    )
    second = atoms.copy()
    second.positions += [0.1, 0.0, 0.0]
    second.info["energy"] = -7.75
    path = tmp_path / "frames.xyz"
    dump_xyz(path, atoms)
    dump_xyz(path, second)

    frames = read_xyz(path)
    ase_frames = read(path, index=":", format="extxyz")
    assert len(frames) == len(ase_frames) == 2
    for expected, actual, ase_frame in zip([atoms, second], frames, ase_frames):
        assert actual.get_chemical_symbols() == expected.get_chemical_symbols()
        assert_allclose(actual.positions, expected.positions)
        assert_allclose(actual.cell, expected.cell)
        assert_array_equal(actual.pbc, expected.pbc)
        assert_allclose(actual.get_masses(), expected.get_masses())
        assert_allclose(actual.get_velocities(), expected.get_velocities())
        assert actual.info["energy"] == pytest.approx(expected.info["energy"])
        assert_allclose(actual.info["forces"], forces)
        assert_allclose(actual.info["stress"], stress)
        assert_array_equal(actual.info["group"], groups)
        assert_allclose(actual.info["mag"], mag)
        assert actual.info["config_type"] == "alloy_bulk"
        assert actual.info["weight"] == pytest.approx(1.25)

        assert_array_equal(ase_frame.numbers, expected.numbers)
        assert_allclose(ase_frame.positions, expected.positions)
        assert_allclose(ase_frame.cell, expected.cell)
        assert_array_equal(ase_frame.pbc, expected.pbc)
        assert ase_frame.get_potential_energy() == pytest.approx(expected.info["energy"])
        assert_allclose(ase_frame.get_stress(voigt=False), stress)
        assert_allclose(ase_frame.info["virial"], -stress * 60.0)
        # Wizard's singular force field is an ASE per-atom array.
        assert_allclose(ase_frame.arrays["force"], forces)
        assert_allclose(ase_frame.arrays["mag"], mag)
        assert_array_equal(ase_frame.arrays["group"], groups.T)


@pytest.mark.parametrize("force_field", ["force", "forces"])
def test_read_independent_virial_fixture(tmp_path, force_field):
    path = tmp_path / "reference.xyz"
    path.write_text(
        '2\n'
        'Lattice="2 0 0 0 3 0 0 0 4" pbc="T F T" '
        'energy=-2.5 virial="24 0 0 0 48 0 0 0 72" '
        f'Properties=pos:R:3:species:S:1:{force_field}:R:3:mass:R:1\n'
        '0.2 0.3 0.4 Fe 1 2 3 55.845\n'
        '1.2 1.3 1.4 Ni -1 -2 -3 58.6934\n',
        encoding="utf-8",
    )

    frames = read_xyz(path)
    assert len(frames) == 1
    atoms = frames[0]
    assert atoms.get_chemical_symbols() == ["Fe", "Ni"]
    assert_allclose(atoms.positions, [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]])
    assert_allclose(atoms.get_masses(), [55.845, 58.6934])
    assert_array_equal(atoms.pbc, [True, False, True])
    assert atoms.get_volume() == pytest.approx(24.0)
    assert atoms.info["energy"] == pytest.approx(-2.5)
    assert_allclose(atoms.info["forces"], [[1, 2, 3], [-1, -2, -3]])
    # Positive virial corresponds to negative ASE stress, in eV/angstrom^3.
    assert_allclose(atoms.info["stress"], np.diag([-1.0, -2.0, -3.0]))


def test_read_ase_written_energy_forces_and_stress(tmp_path):
    atoms = Atoms("Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]], cell=[3.6] * 3, pbc=True)
    forces = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
    stress = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    atoms.calc = SinglePointCalculator(atoms, energy=-5.0, forces=forces, stress=stress)
    path = tmp_path / "ase.xyz"
    write(path, atoms, format="extxyz")

    restored = read_xyz(path)[0]
    assert_array_equal(restored.numbers, atoms.numbers)
    assert_allclose(restored.positions, atoms.positions)
    assert_allclose(restored.cell, atoms.cell)
    assert_allclose(restored.get_masses(), atoms.get_masses())
    assert restored.info["energy"] == pytest.approx(-5.0)
    assert_allclose(restored.info["forces"], forces)
    assert_allclose(restored.info["stress"], np.diag([0.1, 0.2, 0.3]))
