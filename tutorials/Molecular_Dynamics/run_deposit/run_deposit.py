from ase.build import surface
from ase import Atom
from wizard.io import read_xyz
from wizard.atoms import SymbolInfo
from calorine.calculators import CPUNEP
from wizard.atoms import Morph
import numpy as np

symbol_info = SymbolInfo('Cu', 'fcc', 3.631)
calc = CPUNEP('../potentials/nep89_20250409.txt')
atoms = symbol_info.create_bulk_atoms((1,1,1))
slab = surface(atoms, (1,1,1), layers = 24, vacuum=100) 
slab = slab * (8,8,1)
Morph(slab).gpumd(run_in= ['potential nep89_20250409.txt', 
                           'velocity 300', 
                           'time_step 1', 
                           'ensemble npt_mttk temp 300 300 iso 0.0 0.0', 
                           'dump_restart 30000', 
                           'run 30000'],
                  nep_path='../potentials/nep89_20250409.txt')

for i in range(20):
    if i == 0:
        atoms = read_xyz('relax/restart.xyz')[-1]
    else:
        atoms = read_xyz('deposit/{}/restart.xyz'.format(i-1))[-1]

    group = []
    thickness = 5 #A 
    for atom in atoms:
        if atom.position[2] < thickness:
            group.append(0) # fixed group
        else:
            group.append(1)
    atoms.info['group'] = [group]
    cell = atoms.get_cell()
    for _ in range(20):
        a = cell[0]
        b = cell[1]
        u = np.random.rand()
        v = np.random.rand()
        r = u * a + v * b
        
        x = r[0]
        y = r[1]
        z = 60    
        
        atoms.append(Atom('C', position=(x, y, z), momentum=(0, 0, -0.12)))
        atoms.info['group'][0].append(1)

    Morph(atoms).zero_momentum()
    Morph(atoms).gpumd('deposit/{}'.format(i), 
                       run_in=['potential nep89_20250409.txt', 
                                'velocity 300', 
                                'time_step 0.5', 
                                'ensemble nve',
                                'fix 0',
                                'dump_exyz 1000',
                                'run 10000',
                                'time_step 1', 
                                'ensemble nvt_nhc 300 300 100', 
                                'dump_thermo 1000', 
                                'dump_restart 10000', 
                                'run 50000'],
                        nep_path='../potentials/nep89_20250409.txt')