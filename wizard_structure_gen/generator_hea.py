from wizard.atoms import Morph
from wizard.io import dump_xyz
from ase.build import bulk

atoms = bulk('W', 'bcc', 3.1854, cubic = True) * (200, 200, 200)
morph = Morph(atoms)
morph.prop_element_set(['V','Nb','Mo','Ta','W'])
hea = morph.get_atoms()
dump_xyz('example/generator_hea/hea.xyz', atoms)