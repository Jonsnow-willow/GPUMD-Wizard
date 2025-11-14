from wizard.atoms import SymbolInfo
from wizard.atoms import Morph

symbol_infos = [
    SymbolInfo('V',  'bcc', 2.997),
    SymbolInfo('Nb', 'bcc', 3.308),
    SymbolInfo('Mo', 'bcc', 3.163),
    SymbolInfo('Ta', 'bcc', 3.321),
    SymbolInfo('W',  'bcc', 3.185),
    SymbolInfo('VNbMoTaW',  'bcc', 3.195)
    ]

for symbol_info in symbol_infos:
    atoms = symbol_info.create_bulk_atoms((3,3,3))
    dirname = f'{symbol_info.formula}/{symbol_info.lattice_constant}/crystallization'
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
    Morph(atoms).gpumd(dirname=dirname, run_in=run_in, nep_path='../potentials/MoNbTaVW.txt')
