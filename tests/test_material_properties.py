"""Small CPU material-property checks using ASE's built-in Cu EMT potential."""

import re
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from ase.units import GPa
from scipy.optimize import minimize_scalar

from wizard.calc.calculator import MaterialCalculator


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """MaterialCalculator writes its artifacts in the current directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def equilibrium_lattice_constant():
    """An independent scalar E(a) minimum, without Wizard's cell optimizer."""
    def energy(a):
        atoms = bulk("Cu", "fcc", a=a, cubic=True)
        atoms.calc = EMT()
        return atoms.get_potential_energy()

    result = minimize_scalar(
        energy, bounds=(3.4, 3.8), method="bounded", options={"xatol": 1e-10}
    )
    assert result.success
    return result.x


def test_clamped_energy_and_lattice_report(tmp_path):
    a = 3.61
    atoms = bulk("Cu", "fcc", a=a, cubic=True)
    reference = atoms.copy()
    reference.calc = EMT()
    expected_energy = reference.get_potential_energy() / len(reference)

    calculator = MaterialCalculator(atoms, EMT(), clamped=True)
    output = "\n".join(calculator.lattice_constant())

    assert calculator.atom_energy == pytest.approx(expected_energy, abs=1e-12)
    np.testing.assert_allclose(calculator.atoms.cell, np.eye(3) * a, atol=1e-12)
    # Construction must not attach a calculator or relax the caller's structure.
    assert atoms.calc is None
    np.testing.assert_allclose(atoms.positions, reference.positions, atol=1e-12)
    assert f"a={a:.4f} Å  b={a:.4f} Å  c={a:.4f} Å" in output
    assert "α=90.00°  β=90.00°  γ=90.00°" in output
    assert f"V/atom={a**3 / 4:.3f} Å³/atom" in output
    match = re.search(r"Ground_State_Energy:\s+([-\d.]+) eV/atom", output)
    assert match is not None
    # The public report rounds energies to four decimal places.
    assert float(match.group(1)) == pytest.approx(expected_energy, abs=5.1e-5)
    assert (tmp_path / "MaterialProperties.out").read_text() == output + "\n"

    frames = read(tmp_path / "MaterialProperties.xyz", index=":")
    assert len(frames) == 1
    assert frames[0].get_chemical_symbols() == ["Cu"] * 4
    np.testing.assert_allclose(frames[0].cell, reference.cell, atol=1e-12)
    np.testing.assert_allclose(frames[0].positions, reference.positions, atol=1e-8)


def test_cell_relaxation_matches_independent_energy_minimum(
    equilibrium_lattice_constant,
):
    atoms = bulk("Cu", "fcc", a=3.75, cubic=True)
    initial_cell = atoms.cell.copy()
    reference = atoms.copy()
    reference.calc = EMT()
    initial_energy = reference.get_potential_energy() / len(reference)

    calculator = MaterialCalculator(
        atoms, EMT(), hydrostatic_strain=True, fmax=1e-5, steps=100
    )

    np.testing.assert_allclose(
        calculator.atoms.cell.lengths(), equilibrium_lattice_constant, atol=2e-4
    )
    assert calculator.atom_energy < initial_energy
    # Relaxed FCC Cu is stationary with respect to atomic and cell displacements.
    np.testing.assert_allclose(calculator.atoms.get_forces(), 0.0, atol=1e-5)
    np.testing.assert_allclose(calculator.atoms.get_stress(), 0.0, atol=1e-5)
    np.testing.assert_allclose(atoms.cell, initial_cell, atol=1e-12)


def test_elastic_constants_match_energy_curvature(equilibrium_lattice_constant):
    atoms = bulk("Cu", "fcc", a=equilibrium_lattice_constant, cubic=True)
    calculator = MaterialCalculator(atoms, EMT(), clamped=True)
    stiffness = calculator.elastic_constant(epsilon=1e-3)["Cij"]

    # Calorine obtains stiffness from stress. Use total-energy curvature here
    # instead, with engineering shear gamma: F_yz = F_zy = gamma / 2.
    reference = atoms.copy()
    reference.calc = EMT()
    energy0 = reference.get_potential_energy()
    volume = reference.get_volume()
    delta = 1e-3

    def curvature(strain):
        energies = []
        for sign in (-1, 1):
            deformed = atoms.copy()
            deformed.set_cell(atoms.cell @ (np.eye(3) + sign * delta * strain),
                              scale_atoms=True)
            deformed.calc = EMT()
            energies.append(deformed.get_potential_energy())
        return (sum(energies) - 2 * energy0) / (volume * delta**2) / GPa

    c11 = curvature(np.diag([1.0, 0.0, 0.0]))
    # For cubic crystals, d²E/dε² / V = 3 C11 + 6 C12 under isotropic strain.
    c12 = (curvature(np.eye(3)) - 3 * c11) / 6
    shear = np.zeros((3, 3))
    shear[1, 2] = shear[2, 1] = 0.5
    c44 = curvature(shear)

    assert stiffness.shape == (6, 6)
    np.testing.assert_allclose(stiffness, stiffness.T, atol=1e-6)
    np.testing.assert_allclose(
        [stiffness[0, 0], stiffness[0, 1], stiffness[3, 3]],
        [c11, c12, c44], rtol=2e-3, atol=0.05,
    )
    np.testing.assert_allclose(np.diag(stiffness)[:3], stiffness[0, 0], atol=1e-6)
    np.testing.assert_allclose(np.diag(stiffness)[3:], stiffness[3, 3], atol=1e-6)
    assert np.linalg.eigvalsh(stiffness).min() > 0
    np.testing.assert_allclose(calculator.atoms.cell, atoms.cell, atol=1e-12)


def test_eos_artifacts_match_independent_ase_energies(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    calculator = MaterialCalculator(atoms, EMT(), clamped=True)
    figure = Path(calculator.eos_curve())
    table_path = next((tmp_path / "eos_curve_out").glob("*.out"))
    table = np.loadtxt(table_path, skiprows=1)
    frames = read(tmp_path / "MaterialProperties.xyz", index=":")

    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert table.shape == (len(frames), 2)
    assert len(frames) > 10
    assert np.all(np.diff(table[:, 0]) > 0)
    for frame, (volume_per_atom, energy_per_atom) in zip(frames, table):
        frame.calc = EMT()
        # The table rounds volumes to two and energies to four decimal places.
        assert volume_per_atom == pytest.approx(frame.get_volume() / len(frame), abs=0.0051)
        assert energy_per_atom == pytest.approx(
            frame.get_potential_energy() / len(frame), abs=5.1e-5
        )
    # Both compression and expansion must raise the energy around the minimum.
    minimum = np.argmin(table[:, 1])
    assert 0 < minimum < len(table) - 1
    np.testing.assert_allclose(calculator.atoms.cell, atoms.cell, atol=1e-12)
