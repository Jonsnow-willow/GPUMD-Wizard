"""Prepare inspectable GPUMD inputs without an executable or a real potential."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from ase.io import read
from numpy.testing import assert_allclose, assert_array_equal

from wizard.structure.atoms import AlloyInfo, Morph


@pytest.mark.parametrize("include_stopping", [False, True])
def test_gpumd_prepare_without_execution(tmp_path, monkeypatch, include_stopping):
    monkeypatch.chdir(tmp_path)
    execute = Mock(side_effect=AssertionError("run=False must not execute GPUMD"))
    monkeypatch.setattr("wizard.structure.atoms.os.system", execute)
    potential = tmp_path / "nep.txt"
    potential.write_text("input-staging fixture, not a fitted potential\n", encoding="utf-8")
    stopping = tmp_path / "electron_stopping_fit.txt"
    if include_stopping:
        stopping.write_text("0.0 0.0\n1.0 0.1\n", encoding="utf-8")
    atoms = AlloyInfo("Fe", "bcc", 3.0).create_bulk_atoms((2, 2, 2))
    before = atoms.copy()
    run_in = ["potential nep.txt", "time_step 1", "dump_exyz 1", "run 0"]
    destination = tmp_path / "prepared"

    Morph(atoms).gpumd(
        dirname=str(destination),
        run_in=run_in,
        nep_path=str(potential),
        electron_stopping_path=str(stopping),
        run=False,
    )

    execute.assert_not_called()
    assert Path.cwd() == tmp_path
    assert (destination / "run.in").read_text(encoding="utf-8") == "\n".join(run_in) + "\n"
    assert (destination / potential.name).read_bytes() == potential.read_bytes()
    assert (destination / stopping.name).exists() == include_stopping
    if include_stopping:
        assert (destination / stopping.name).read_bytes() == stopping.read_bytes()
    model = read(destination / "model.xyz", format="extxyz")
    assert_array_equal(model.numbers, before.numbers)
    assert_allclose(model.positions, before.positions)
    assert_allclose(model.cell, before.cell)
    assert_array_equal(model.pbc, before.pbc)
    assert model.info["config_type"] == "bcc_bulk"
    assert_allclose(atoms.positions, before.positions)
    assert_allclose(atoms.cell, before.cell)
    assert atoms.info == before.info


def test_gpumd_does_not_overwrite_an_existing_run(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "run.in"
    marker.write_text("existing simulation input\n", encoding="utf-8")
    atoms = AlloyInfo("Fe", "bcc", 3.0).create_bulk_atoms((1, 1, 1))

    with pytest.raises(FileExistsError):
        Morph(atoms).gpumd(dirname=str(destination), run=False)

    assert marker.read_text(encoding="utf-8") == "existing simulation input\n"
