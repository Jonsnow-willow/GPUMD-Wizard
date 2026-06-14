from wizard.structure.atoms import AlloyInfo
from wizard.structure.atoms import Morph

alloy_infos = [
    AlloyInfo('V',  'bcc', 2.997),
    AlloyInfo('Nb', 'bcc', 3.308),
    AlloyInfo('Mo', 'bcc', 3.163),
    AlloyInfo('Ta', 'bcc', 3.321),
    AlloyInfo('W',  'bcc', 3.185),
    AlloyInfo('VNbMoTaW',  'bcc', 3.195)
    ]

for alloy_info in alloy_infos:
    atoms = alloy_info.create_bulk_atoms((3,3,3))
    dirname = f'{alloy_info.formula}/{alloy_info.lattice_constant}/crystallization'
    run_in = ['potential nep.txt', 
              'velocity 2000', 
              'time_step 1', 
              'ensemble npt_scr 2000 2000 200 0 500 2000', 
              'dump_thermo 1000', 
              'dump_exyz 10000', 
              'dump_restart 10000', 
              'run 100000',
              'ensemble npt_scr 2000 5000 200 0 500 2000',
              'dump_thermo 1000', 
              'dump_exyz 100000', 
              'dump_restart 10000', 
              'run 10000000',
              'ensemble npt_scr 4500 4500 200 0 500 2000',
              'dump_thermo 1000', 
              'dump_exyz 10000', 
              'dump_restart 10000', 
              'run 100000',
              'ensemble npt_scr 4500 1500 200 0 500 2000',
              'dump_thermo 1000', 
              'dump_exyz 100000', 
              'dump_restart 10000', 
              'run 10000000']
    Morph(atoms).gpumd(dirname=dirname, run_in=run_in, nep_path='../potentials/nep.txt')
