from wizard.generator import Generator
from wizard.frames import MultiMol
from calorine.calculators import CPUNEP

new_elements = ['Nb']
ex_elements = ['Mo', 'Ta', 'W', 'V']

generator = Generator(elements = new_elements, symbols=ex_elements, 
                      lattice_types = ['fcc','bcc','hcp'], adjust_ratios = 4)
calc = CPUNEP('../potentials/nep89_20250409.txt')
frames = generator.get_bulk_structures(calc=calc,
                                       supercell={'bcc':(3,3,3),'fcc':(3,3,3),'hcp':(3,3,3)})
MultiMol(frames).dump('train.xyz')