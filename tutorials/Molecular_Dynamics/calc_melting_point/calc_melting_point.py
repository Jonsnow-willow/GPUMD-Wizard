from wizard.atoms import SymbolInfo
from wizard.atoms import Morph
from wizard.io import read_xyz

symbol_info = SymbolInfo('W',  'bcc', 3.185)
atoms = symbol_info.create_bulk_atoms((10,10,20))

group = []
for atom in atoms:
    if atom.position[2] < atoms.cell[2, 2] / 2:
        group.append(0)
    else:
        group.append(1)
atoms.info['group'] = [group]

run_in_1 = ['potential nep.txt', 
            'velocity 3000', 
            'time_step 1', 
            'ensemble npt_ber 3000 3000 200 0 500 2000', 
            'dump_exyz 10000', 
            'dump_thermo 1000',
            'run 30000',
            'ensemble heat_lan 3500 200 500 0 1',
            'dump_exyz 10000',
            'dump_thermo 1000',
            'dump_restart 10000',
            'run 1000000']

Morph(atoms).gpumd('melting_point/relax', run_in_1, nep_path='../potentials/MoNbTaVW.txt')

for Tm in range(3400, 3701, 100):
    atoms = read_xyz('melting_point/relax/restart.xyz')[-1]
    run_in = ['potential nep.txt', 
             f'velocity {Tm}', 
              'time_step 1', 
             f'ensemble npt_ber {Tm} {Tm} 200 0 500 2000', 
              'dump_exyz 10000', 
              'dump_thermo 1000',
              'run 30000']
    Morph(atoms).gpumd(f'melting_point/{Tm}', run_in, nep_path='../potentials/MoNbTaVW.txt')