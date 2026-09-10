# CPU test suite

The suite currently defines 25 test cases, including parameterized cases. It covers public data exchange, structure operations, GPUMD input preparation, and small material-property calculations using ASE's built-in Cu EMT potential. All generated structures, tables, and figures are written to pytest temporary directories.

From the repository root, in an isolated Python 3.10 or newer environment:

```bash
python -m pip install '.[test]'
python -m pytest -q
```

| File | What it checks | Independent reference |
| --- | --- | --- |
| `test_io.py` | XYZ property-column slices, absent fields, force aliases, symbol normalization, and invalid magnetic-moment widths | Small explicit property strings and atom rows with known expected fields; contributed in PR #8 |
| `test_io_roundtrip.py` | Extended-XYZ frame append, geometry, selected metadata, forces, magnetic moments, and stress/virial conversion | ASE reading and writing, plus a hand-written virial fixture with a known cell volume and stress |
| `test_structure.py` | Crystal sizes and volumes, exactly representable alloy composition, interstitial insertion, and Wigner-Seitz defect counts under periodic wrapping and affine strain | Crystallographic atom counts and volumes, specified defect locations, and atom/site conservation |
| `test_gpumd_inputs.py` | `Morph.gpumd(run=False)` file preparation, optional electron-stopping input, and preservation of an existing run directory | Exact input text and copied bytes, ASE reading of `model.xyz`, and a mocked executable call that fails if invoked |
| `test_material_properties.py` | Clamped energy and lattice reporting, relaxation, elastic constants, and equation-of-state outputs | Direct ASE EMT evaluations, a scalar energy minimum, and total-energy curvature for comparison with stress-derived elastic constants |

The configured CI matrix uses Python 3.10 and 3.13. It builds and installs a wheel, then runs the tests outside the source checkout with Python's isolated mode (`-I`) to check the installed package without a developer's `PYTHONPATH`.

GPUMD execution, NEP potential evaluation, phonon calculations, migration paths, and long MD/MC simulations require separate manual validation with their relevant inputs and dependencies. Record the revision, commands, potential files, convergence settings, and results for each manual validation. The selected outputs committed with tutorials provide examples of those workflows.
