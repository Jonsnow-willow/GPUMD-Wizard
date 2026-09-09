import numpy as np
import pytest
from ase import Atoms

from wizard.structure.analysis import find_asd, find_msd, unwrap, wigner_seitz


def lattice(positions, cell=10.0, symbols=None):
    n = len(positions)
    return Atoms(
        symbols=symbols or ['Fe'] * n,
        positions=positions,
        cell=[cell, cell, cell],
        pbc=[True, True, True],
    )


def drifting_frames(step=1.0, n_frames=4):
    """One atom moving a fixed distance along x each frame, no wrapping."""
    return [lattice([(i * step, 0.0, 0.0)]) for i in range(n_frames)]


def test_wigner_seitz_rejects_non_atoms_arguments():
    ref = lattice([(0.0, 0.0, 0.0)])
    with pytest.raises(TypeError):
        wigner_seitz('not atoms', ref)
    with pytest.raises(TypeError):
        wigner_seitz(ref, 'not atoms')


def test_wigner_seitz_rejects_a_non_positive_chunk_size():
    ref = lattice([(0.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match='positive integer'):
        wigner_seitz(ref, ref, chunk_size=0)


def test_wigner_seitz_assigns_each_atom_to_its_nearest_site():
    ref = lattice([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (5.0, 5.0, 0.0)])
    atoms = lattice([(0.1, 0.0, 0.0), (5.1, 0.0, 0.0), (0.0, 5.1, 0.0), (5.0, 5.1, 0.0)])
    result = wigner_seitz(atoms, ref)
    assert list(result['atom_sites']) == [0, 1, 2, 3]
    assert result['atom_distances'] == pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert list(result['vacancies']) == []
    assert list(result['interstitials']) == []


def test_wigner_seitz_result_is_independent_of_chunk_size():
    ref = lattice([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (5.0, 5.0, 0.0)])
    atoms = lattice([(0.1, 0.0, 0.0), (5.1, 0.0, 0.0), (0.0, 5.1, 0.0), (5.0, 5.1, 0.0)])
    chunked, whole = wigner_seitz(atoms, ref, chunk_size=1), wigner_seitz(atoms, ref)
    assert chunked.keys() == whole.keys()
    for key in whole:
        assert np.asarray(chunked[key], dtype=object).tolist() == \
               np.asarray(whole[key], dtype=object).tolist()


def test_wigner_seitz_finds_a_vacancy_and_an_interstitial():
    # Two sites 5 A apart with both atoms next to the first one. Site 0 keeps one
    # atom as its occupant and reports the other as an interstitial; site 1 is empty.
    ref = lattice([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    atoms = lattice([(0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)])
    result = wigner_seitz(atoms, ref)
    assert list(result['vacancies']) == [1]
    assert list(result['interstitials']) == [0]
    assert list(result['interstitial_sites']) == [0]
    assert list(result['occupants']) == [1, -1]
    assert [list(s) for s in result['site_atoms']] == [[0, 1], []]
    assert list(result['atom_sites']) == [0, 0]


def test_unwrap_removes_a_jump_across_the_periodic_boundary():
    # The atom steps from x=9.5 to x=0.5 in a 10 A cell, which is a +1 A move
    # across the boundary rather than the -9 A move the raw coordinates imply.
    frames = [lattice([(9.5, 0.0, 0.0)]), lattice([(0.5, 0.0, 0.0)])]
    unwrapped = unwrap(frames)
    assert unwrapped[1].positions[0][0] == pytest.approx(10.5)


def test_unwrap_leaves_a_trajectory_without_jumps_alone():
    frames = drifting_frames(step=1.0)
    expected = [f.positions.copy() for f in frames]
    for frame, before in zip(unwrap(frames), expected):
        assert frame.positions == pytest.approx(before)


def test_find_asd_grows_as_the_square_of_the_displacement():
    # Displacement from frame 0 is 0, 1, 2, 3 A, so the squared values are 0, 1, 4, 9.
    assert find_asd(drifting_frames(step=1.0)) == pytest.approx([0.0, 1.0, 4.0, 9.0])


def test_find_asd_reports_per_symbol_and_average_when_symbols_are_given():
    frames = [
        lattice([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], symbols=['Fe', 'Cr']),
        lattice([(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)], symbols=['Fe', 'Cr']),
    ]
    asd = find_asd(frames, 'Fe', 'Cr')
    assert asd['Fe'] == pytest.approx([0.0, 1.0])
    assert asd['Cr'] == pytest.approx([0.0, 0.0])
    # The average is over both atoms, so half of the Fe-only value.
    assert asd['average'] == pytest.approx([0.0, 0.5])


def test_find_msd_averages_over_time_origins():
    # Four frames 1 A apart. Lag 1 has three origins each contributing 1,
    # lag 2 has two origins each contributing 4, lag 3 has one contributing 9.
    assert find_msd(drifting_frames(step=1.0)) == pytest.approx([0.0, 1.0, 4.0, 9.0])


def test_find_msd_caps_the_lag_count_at_the_number_of_frames():
    frames = drifting_frames(step=1.0, n_frames=3)
    assert len(find_msd(frames, Nc=100)) == 3
    assert len(find_msd(frames, Nc=2)) == 2


def test_find_msd_reports_per_symbol_and_average_when_symbols_are_given():
    frames = [
        lattice([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], symbols=['Fe', 'Cr']),
        lattice([(2.0, 0.0, 0.0), (0.0, 0.0, 0.0)], symbols=['Fe', 'Cr']),
    ]
    msd = find_msd(frames, 2, 'Fe', 'Cr')
    assert msd['Fe'] == pytest.approx([0.0, 4.0])
    assert msd['Cr'] == pytest.approx([0.0, 0.0])
    assert msd['average'] == pytest.approx([0.0, 2.0])
    assert np.all(np.diff(msd['Fe']) >= 0)
