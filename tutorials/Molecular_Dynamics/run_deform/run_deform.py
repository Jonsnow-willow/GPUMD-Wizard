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

temperature = 300 #K
strain_rate = 2e8
dt = 1e-15 #fs
for symbol_info in symbol_infos:
    atoms = symbol_info.create_bulk_atoms((3,3,3))
    dirname = f'{symbol_info.formula}/{symbol_info.lattice_type}/{temperature}K/utc_tensile'
    length = atoms.cell[2, 2]
    strain = strain_rate * dt * length
    run_in = ['potential nep.txt', 
              'velocity 300', 
              'time_step 1',
              f'ensemble npt_scr {temperature} {temperature} 200 0 500 2000',
              'run 30000'
              f'ensemble npt_scr {temperature} {temperature} 100 0 0 0 100 100 100 1000',
              f'deform {strain} 0 0 1', 
              'dump_thermo 1000', 
              'dump_exyz 200000', 
              'dump_restart 10000',
              'run 2000000']
    Morph(atoms).gpumd(dirname=dirname, run_in=run_in, nep_path='../potentials/MoNbTaVW.txt')
