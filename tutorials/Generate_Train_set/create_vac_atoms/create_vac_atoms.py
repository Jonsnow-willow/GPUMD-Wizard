from wizard.frames import MultiMol
from wizard.io import read_xyz
from calorine.calculators import CPUNEP

init = read_xyz('../create_init_alloys/train.xyz')
calc = CPUNEP('../potentials/nep89_20250409.txt')
vacancies = MultiMol(init).get_vacancies(num=[1,3,6])
MultiMol(vacancies).dump('train.xyz')


