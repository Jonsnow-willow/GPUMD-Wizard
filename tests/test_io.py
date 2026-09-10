import pytest

from wizard.utils.io import (
    _parsed_properties,
    _read_force,
    _read_group,
    _read_mag,
    _read_mass,
    _read_positions,
    _read_symbols,
)


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
