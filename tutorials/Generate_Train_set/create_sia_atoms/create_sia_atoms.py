from wizard.frames import MultiMol

from wizard.io import read_xyz
from calorine.calculators import CPUNEP

init = read_xyz('../create_init_alloys/train.xyz')
calc = CPUNEP('../potentials/nep89_20250409.txt')
vectors = [(1,1,1),(1,0,0),(1,1,0)]
sias = MultiMol(init).get_sias(calc, vectors)
MultiMol(sias).dump('train.xyz')


