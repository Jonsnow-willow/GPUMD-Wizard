from ase.build import bulk
from wizard.io import read_xyz
from wizard.atoms import Morph
import numpy as np

lc = 3.1854
duplicate = [50, 50, 50]
pka_energy = 1000
angle = np.array([1, 3, 5])    
index = int(2 * (duplicate[0] * duplicate[1] * duplicate[2] / 2 + duplicate[0] * duplicate[1] / 2 + duplicate[0] / 2))

group = []
for i in range(duplicate[0]):
    for j in range(duplicate[1]):
        for k in range(duplicate[2]):
            if i < 3 or j < 3 or k < 3:
                group.append(0)
            elif i >= duplicate[0] - 3 or j >= duplicate[1] - 3 or k >= duplicate[2] - 3:
                group.append(1)
            else:
                group.append(2)
            
for i in range (10):
    initial = bulk('W', 'bcc', lc, cubic = True) * (duplicate, duplicate, duplicate)
    initial.info['group'] = group
    Morph(initial).gpumd(f'radiation{i}' + '/relax', 
                         run_in= ['potential ../../hea.txt', 'velocity 300', 'time_step 1', 
                                  'ensemble npt_scr 300 300 200 0 500 2000', 
                                  'dump_thermo 1000', 'dump_restart 30000', 'run 30000'])
    atoms = read_xyz(f'radiation{i}' + '/relax/restart.xyz')[0]
    Morph(atoms).set_pka(pka_energy, angle, index)
    Morph(atoms).gpumd(f'radiation{i}' + '/cascade',
                       run_in= ['potential ../../hea.txt', 
                                'velocity 300', 'time_step 0', 
                                'ensemble nve','dump_exyz 1 1 1', 'run 1',
                                'time_step 1 0.015', 
                                'ensemble heat_nhc 300 200 0 0 1',
                                'electron_stop ../../electron_stopping_fit.txt',
                                'compute 0 200 10 temperature', 'dump_restart 10000', 
                                'dump_exyz 2000 1 1','run 40000',
                                'ensemble heat_nhc 300 200 0 0 1',
                                'electron_stop ../../electron_stopping_fit.txt',
                                'compute 0 500 10 temperature', 'dump_restart 10000', 
                                'dump_exyz 5000 1 1','run 40000'])
                       

