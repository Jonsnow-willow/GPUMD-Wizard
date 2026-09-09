import numpy as np
import pytest
from ase import Atoms

from wizard.utils.io import (
    _parsed_properties,
    _read_force,
    _read_group,
    _read_mag,
    _read_mass,
    _read_positions,
    _read_symbols,
    dump_xyz,
    read_xyz,
    write_run,
)


def make_atoms():
    return Atoms(
        symbols=['Pb', 'Te'],
        positions=[(0.0, 0.0, 0.0), (1.5, 1.5, 1.5)],
        cell=[3.0, 3.0, 3.0],
        pbc=[True, True, True],
    )


def test_write_run_writes_one_parameter_per_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_run(['potential nep.txt', 'velocity 300', 'ensemble nvt_ber 300 300 100'])
    assert (tmp_path / 'run.in').read_text().splitlines() == [
        'potential nep.txt',
        'velocity 300',
        'ensemble nvt_ber 300 300 100',
    ]


def test_parsed_properties_maps_each_name_to_its_column_slice():
    comment = 'properties=species:s:1:pos:r:3:mass:r:1:force:r:3'
    assert _parsed_properties(comment) == {
        'species': slice(0, 1),
        'pos': slice(1, 4),
        'mass': slice(4, 5),
        'force': slice(5, 8),
    }


def test_read_helpers_pull_the_right_columns():
    props = _parsed_properties(
        'properties=species:s:1:pos:r:3:mass:r:1:force:r:3:group:i:2:mag:r:1'
    )
    words = ['pb', '0.0', '1.0', '2.0', '207.2', '0.1', '0.2', '0.3', '0', '1', '0.5']
    assert _read_symbols(words, props) == 'Pb'
    assert _read_positions(words, props) == (0.0, 1.0, 2.0)
    assert _read_mass(words, props) == pytest.approx(207.2)
    assert _read_force(words, props) == pytest.approx((0.1, 0.2, 0.3))
    assert _read_group(words, props) == [0, 1]
    assert _read_mag(words, props) == pytest.approx(0.5)


def test_read_helpers_return_none_for_absent_properties():
    props = _parsed_properties('properties=species:s:1:pos:r:3')
    words = ['pb', '0.0', '1.0', '2.0']
    assert _read_mass(words, props) is None
    assert _read_force(words, props) is None
    assert _read_group(words, props) is None
    assert _read_mag(words, props) is None


def test_read_force_accepts_either_forces_or_force_as_the_column_name():
    words = ['pb', '0.0', '0.0', '0.0', '1.0', '2.0', '3.0']
    for name in ('force', 'forces'):
        props = _parsed_properties(f'properties=species:s:1:pos:r:3:{name}:r:3')
        assert _read_force(words, props) == pytest.approx((1.0, 2.0, 3.0))


def test_read_mag_rejects_a_component_count_it_cannot_interpret():
    props = _parsed_properties('properties=species:s:1:pos:r:3:mag:r:2')
    words = ['pb', '0.0', '0.0', '0.0', '0.5', '0.6']
    with pytest.raises(ValueError, match='1 or 3 components'):
        _read_mag(words, props)


def test_round_trip_preserves_species_positions_cell_and_masses(tmp_path):
    atoms = make_atoms()
    path = tmp_path / 'model.xyz'
    dump_xyz(str(path), atoms)

    (back,) = read_xyz(str(path))
    assert back.get_chemical_symbols() == ['Pb', 'Te']
    assert back.positions == pytest.approx(atoms.positions)
    assert np.asarray(back.get_cell()) == pytest.approx(np.asarray(atoms.get_cell()))
    assert list(back.get_pbc()) == [True, True, True]
    assert back.get_masses() == pytest.approx(atoms.get_masses())


def test_round_trip_preserves_energy_forces_and_velocities(tmp_path):
    atoms = make_atoms()
    atoms.info['energy'] = -12.5
    atoms.info['forces'] = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    atoms.set_velocities([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    path = tmp_path / 'model.xyz'
    dump_xyz(str(path), atoms)

    (back,) = read_xyz(str(path))
    assert back.info['energy'] == pytest.approx(-12.5)
    assert np.asarray(back.info['forces']) == pytest.approx(atoms.info['forces'])
    assert back.get_velocities() == pytest.approx(atoms.get_velocities())


def test_round_trip_preserves_group_and_mag_columns(tmp_path):
    atoms = make_atoms()
    atoms.info['group'] = [np.array([0, 1]), np.array([1, 1])]
    atoms.info['mag'] = [0.5, -0.5]
    path = tmp_path / 'model.xyz'
    dump_xyz(str(path), atoms)

    (back,) = read_xyz(str(path))
    assert [list(col) for col in back.info['group']] == [[0, 1], [1, 1]]
    assert np.asarray(back.info['mag']).reshape(-1) == pytest.approx([0.5, -0.5])


def test_pbc_false_survives_the_round_trip(tmp_path):
    atoms = make_atoms()
    atoms.set_pbc([True, False, True])
    path = tmp_path / 'model.xyz'
    dump_xyz(str(path), atoms)

    (back,) = read_xyz(str(path))
    assert list(back.get_pbc()) == [True, False, True]


def test_dump_appends_so_repeated_calls_build_a_trajectory(tmp_path):
    path = tmp_path / 'traj.xyz'
    first = make_atoms()
    second = make_atoms()
    second.positions += 0.25
    dump_xyz(str(path), first)
    dump_xyz(str(path), second)

    frames = read_xyz(str(path))
    assert len(frames) == 2
    assert frames[1].positions == pytest.approx(second.positions)


def test_dump_rejects_mag_that_does_not_cover_every_atom(tmp_path):
    atoms = make_atoms()
    atoms.info['mag'] = [0.5]
    with pytest.raises(ValueError, match='Mag data dimensions'):
        dump_xyz(str(tmp_path / 'bad.xyz'), atoms)


def test_dump_rejects_group_that_does_not_cover_every_atom(tmp_path):
    atoms = make_atoms()
    atoms.info['group'] = [np.array([0])]
    with pytest.raises(ValueError, match='Group data dimensions'):
        dump_xyz(str(tmp_path / 'bad.xyz'), atoms)
