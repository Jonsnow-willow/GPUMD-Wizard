from wizard.utils.frames import MultiMol
from wizard.model.atoms import Morph
from calorine.tools import relax_structure
from wizard.utils.io import read_xyz
from calorine.calculators import CPUNEP

fks = []
init = read_xyz('../create_init_alloys/train.xyz')
calc = CPUNEP('../potentials/nep89_20250409.txt')
for atoms in init:
    atoms_copy = atoms.copy()
    Morph(atoms_copy).create_fks(10)
    atoms_copy.calc = calc
    relax_structure(atoms_copy, fmax = 0.05)
    fks.append(atoms_copy)
MultiMol(fks).dump('train.xyz')


