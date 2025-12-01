from wizard.atoms import SymbolInfo
from wizard.atoms import Morph
from wizard.io import read_xyz
import numpy as np

symbol_info = SymbolInfo('W',  'bcc', 3.185)
atoms = symbol_info.create_bulk_atoms((20,20,20))
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
atoms.info['group'] = [group]

run_in_1 = ['potential nep.txt',
            'velocity 300', 
            'time_step 1', 
            'ensemble npt_scr 300 300 200 0 500 2000', 
            'dump_thermo 1000', 
            'dump_restart 30000', 
            'run 30000']

run_in_2 = ['potential nep.txt', 
            'velocity 300', 
            'time_step 0', 
            'ensemble nve',
            'dump_exyz 1', 
            'run 1',
            'time_step 1 0.015', 
            'ensemble heat_nhc 300 200 0 0 1',
            'compute 0 200 10 temperature', 
            'dump_restart 10000', 
            'dump_exyz 2000 1 1',
            'run 70000']

pka_energy = 500 #eV
direction = np.array([1, 3, 5]) 

Morph(atoms).gpumd('radiation/relax', run_in_1, nep_path='../potentials/MoNbTaVW.txt')
atoms = read_xyz('radiation/relax/restart.xyz')[-1]
Morph(atoms).set_pka(pka_energy, direction)
Morph(atoms).gpumd('radiation/cascade', run_in_2, nep_path='../potentials/MoNbTaVW.txt')