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

run_in = ['potential nep.txt', 
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

Morph(atoms).gpumd(nep_path='../potentials/nep.txt')
atoms = read_xyz('relax/restart.xyz')[-1]
Morph(atoms).set_pka(energy=pka_energy, direction=direction)
Morph(atoms).gpumd('radiation/cascade', run_in, nep_path='../potentials/nep.txt')