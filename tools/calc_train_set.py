from wizard.io import relax
from wizard.io import read_xyz
from pynep.calculate import NEP

frames = read_xyz('train.xyz')
energy = []
for atoms in frames:
    atoms.calc = NEP('nep.txt')
    relax(atoms)
    e = atoms.get_potential_energy()
    energy.append(e)

print(energy)
    
