from ase.build import bulk
from wizard.io import dump_xyz
from wizard.atoms import Morph

atoms = bulk('Fe','fcc', a = 3.5, cubic = True) * [5, 5, 5]  
sym = ['Fe'] * 343 + ['Cr'] * 107 + ['Ni'] * 50
atoms.set_chemical_symbols(sym)
conf = Morph(atoms)
conf.shuffle_symbols()  
conf.create_fks(20)
final = conf.get_atoms()
dump_xyz('example/set_fks/dump.xyz', final)
