from wizard.frames import MultiMol
from wizard.io import read_xyz
import numpy as np

init = read_xyz('../create_init_alloys/train.xyz')

scale = np.arange(0.95, 1.06, 0.05)
init_scale = MultiMol(init).deform(scale)

strain_ratio = 0.04
init_scale_strain = MultiMol(init_scale).random_strain(strain_ratio)

max_displacement = 0.4
init_scale_strain_disp = MultiMol(init_scale_strain).random_displacement(max_displacement)

MultiMol(init_scale_strain_disp).dump('rattled_train.xyz')
