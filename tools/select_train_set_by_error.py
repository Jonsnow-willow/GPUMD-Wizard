from pynep.io import load_nep, dump_nep
from pynep.calculate import NEP
from wizard.frames import MultiMol

frames = load_nep('train.xyz', ftype= "exyz")
calc = NEP('nep.txt')
for atoms in frames:
    atoms.calc = calc
mol = MultiMol(frames)
select = mol.select_set_by_error(0.05, 0.07, 1000)
dump_nep('select.xyz', select, ftype = "exyz")
