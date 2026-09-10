# Testing and reproducing a small material-property workflow

The automated tests use small CPU calculations and temporary output directories.
They check structure generation, XYZ data exchange, and selected material-property
interfaces. The Cu property tests use ASE's built-in EMT calculator and compare
against independent ASE energies, an energy-versus-lattice-constant minimum, and
elastic energy curvatures. These tests do not establish the accuracy of every
potential or every material-property workflow.

## Install and run the automated tests

Start in the repository root with Python 3.10 or newer. Create a separate virtual
environment so the dependencies do not change an existing simulation environment:

```bash
python3 -m venv /tmp/gpumd-wizard-validation-venv
source /tmp/gpumd-wizard-validation-venv/bin/activate
python -m pip install -e '.[test]'
python -I -m pytest
```

On Windows, activate the environment using its `Scripts` directory instead.
The `test` extra installs pytest; the package dependencies include ASE, calorine,
phonopy, NumPy, Matplotlib, and spglib. The tests require no GPU, GPUMD executable,
LAMMPS installation, or separately downloaded potential. Run just the material
property checks with:

```bash
python -I -m pytest tests/test_material_properties.py
```

The `-I` option prevents unrelated `PYTHONPATH` settings from overriding the
package installed in this environment.

## Manual CPU example: vanadium with the bundled NEP potential

This is a small part of the existing
[`MoNbTaVW` property tutorial](Calculatiing_Material_Properties/MoNbTaVW/calc_properties.py):
one conventional bcc V cell (two atoms), cell relaxation, lattice and energy
reporting, and an elastic tensor. It uses calorine's `CPUNEP` and the checked-in
`MoNbTaVW/potentials/nep.txt` potential. It does not require the GPUMD executable.

With the environment above active, run the following **from the repository
root**. The script resolves the potential path before changing directories and
copies it into a newly created temporary directory. `MaterialProperties.out` and
`MaterialProperties.xyz` are written there, preserving the tutorial's historical
outputs. The temporary directory is printed and retained for inspection.

```bash
python -I - <<'PY'
from importlib.metadata import version
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
from calorine.calculators import CPUNEP
from wizard.calc.calculator import MaterialCalculator
from wizard.structure.atoms import AlloyInfo

repo = Path.cwd().resolve()
potential = (
    repo / "tutorials/Calculatiing_Material_Properties/MoNbTaVW/potentials/nep.txt"
)
if not potential.is_file():
    raise FileNotFoundError("Run this example from the GPUMD-Wizard repository root.")

workdir = Path(tempfile.mkdtemp(prefix="gpumd-wizard-v-")).resolve()
local_potential = workdir / "nep.txt"
shutil.copy2(potential, local_potential)
os.chdir(workdir)

atoms = AlloyInfo("V", "bcc", 2.997).create_bulk_atoms((1, 1, 1))
properties = MaterialCalculator(
    atoms,
    CPUNEP(str(local_potential)),
    fmax=1e-5,
    steps=100,
    hydrostatic_strain=True,
)
print("\n".join(properties.lattice_constant()))
elastic = properties.elastic_constant(epsilon=1e-3)
print("\n".join(elastic["output"]))

cell = properties.atoms.cell
C = elastic["Cij"]
assert len(properties.atoms) == 2
np.testing.assert_allclose(cell.lengths(), cell.lengths()[0], atol=1e-5)
np.testing.assert_allclose(cell.angles(), 90.0, atol=1e-5)
np.testing.assert_allclose(C, C.T, atol=1e-4)
assert np.linalg.eigvalsh(C).min() > 0

print("Output directory:", workdir)
for package in ("gpumd-wizard", "ase", "calorine", "numpy", "phonopy"):
    print(f"{package}: {version(package)}")
PY
```

The expected files are `nep.txt`, `MaterialProperties.out`, and
`MaterialProperties.xyz`. The XYZ file contains two frames: the relaxed cell
reported by `lattice_constant()` and the reference structure written by
`elastic_constant()`. Both should contain two V atoms with periodic boundaries.

The example was reproduced on 2026-09-10 from a wheel built from the current
checkout, using isolated Python execution. Its environment was Python 3.13.2,
GPUMD-Wizard 2.0.0, ASE 3.29.0, calorine 3.5, NumPy 2.3.0, and phonopy 4.5.0.
The two output frames were also read independently with ASE and verified to have
two V atoms and fully periodic boundaries each.

| Quantity | Reproduced V reference | Comparison tolerance |
| --- | ---: | ---: |
| Conventional cubic lattice constant | 2.9967 Å | 0.003 Å |
| Energy per atom | −8.9915 eV/atom | 0.001 eV/atom |
| C11 = C22 = C33 | 284.54 GPa | 1% or 0.5 GPa, whichever is larger |
| C12 = C13 = C23 | 154.77 GPa | 1% or 0.5 GPa, whichever is larger |
| C44 = C55 = C66 | 22.62 GPa | 1% or 0.5 GPa, whichever is larger |

These tolerances allow for the rounded output, relaxation thresholds,
and numerical differentiation. They are numerical comparison tolerances, not
uncertainties in the potential's physical accuracy. A larger discrepancy should
be investigated by checking the potential file, package versions, cell geometry,
and relaxation convergence. The values are specific to this NEP potential and
must not be compared with the EMT Cu tests as if the calculators or materials
were interchangeable.

For provenance, the bundled potential used for this run has SHA-256
`8fe9ffa43a1675ec4d6b657ec463da7aefa8d2752bee912e1d371c3b038be1d9`.
The unchanged historical
[`MaterialProperties.out`](Calculatiing_Material_Properties/MoNbTaVW/MaterialProperties.out)
reports a lattice constant of 2.9970 Å and C11/C12/C44 of
281.68/153.16/22.41 GPa. Its complete execution environment is not recorded, and
the elastic values differ from this rerun by approximately 1%; the table above
is the baseline for the documented procedure and environment.

## Larger tutorial runs

The full `MoNbTaVW/calc_properties.py` script also calculates phonons, defects,
migration barriers, and alloy properties. It uses larger supercells and takes
longer than the two-atom example above; it is not part of the quick test command.

The existing
[`EAM_Zhou` tutorial](Calculatiing_Material_Properties/EAM_Zhou/calc_properties.py)
uses ASE's `LAMMPSlib`, so it additionally needs a working LAMMPS Python module
and shared library with the `eam/alloy` pair style. Installing GPUMD-Wizard alone
does not provide that external solver. The complete script loops over multiple
metals and includes surfaces, phonons, vacancies, and interstitials; allow for
substantially more work than the CPU smoke tests. Run it from a separate copy of
that tutorial directory to preserve the checked-in reference files and relative
`potentials/` paths. Its recorded numbers use EAM potentials and have their own
physical assumptions and convergence settings.
