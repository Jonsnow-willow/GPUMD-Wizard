from wizard.atoms import Morph, SymbolInfo
from wizard.io import read_xyz
import numpy as np

run_in =['potential nep.txt', 
         'velocity 300', 
         'time_step 1', 
         'ensemble npt_mttk temp 300 300 iso 0 0', 
         'dump_thermo 1000', 
         'dump_restart 10000', 
         'dump_exyz 10000', 
         'run 10000',
         'time_step 1 0.015', 
         'ensemble heat_nhc 300 200 0 0 1',
         'compute 0 100 10 temperature', 
         'dump_restart 10000', 
         'dump_exyz 5000 1 1',
         'run 20000']

pka_energy = 10000
for i in range(1000):
    if i == 0:
        symbol_info = SymbolInfo('W',  'bcc', 3.185)
        atoms = symbol_info.create_bulk_atoms((20,20,20))
    else:
        atoms = read_xyz(f'{i-1}/restart.xyz')[-1]
    Morph(atoms).random_center()
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.arccos(np.random.uniform(0, 1)) 
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    group = []
    thickness_angstrom = 10 #A 
    cell_lengths = np.linalg.norm(atoms.get_cell(), axis=1)
    thickness_frac = thickness_angstrom / cell_lengths  
    scaled_positions = atoms.get_scaled_positions()
    for sp in scaled_positions:
        if (sp < thickness_frac).any():
            group.append(0)
        elif (sp > 1 - thickness_frac).any():
            group.append(1)
        else:
            group.append(2)
    atoms.info['group'] = group
    
    Morph(atoms).set_pka(energy=pka_energy, direction=direction)
    Morph(atoms).gpumd(f'{i}', run_in, nep_path='../potentials/MoNbTaVW.txt')
    
    
