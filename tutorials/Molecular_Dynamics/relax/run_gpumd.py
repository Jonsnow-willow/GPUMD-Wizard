from wizard.atoms import SymbolInfo
from wizard.atoms import Morph

symbol_infos = [
    SymbolInfo('V',  'bcc', 2.997),
    SymbolInfo('Nb', 'bcc', 3.308),
    SymbolInfo('Mo', 'bcc', 3.163),
    SymbolInfo('Ta', 'bcc', 3.321),
    SymbolInfo('W',  'bcc', 3.185),
    SymbolInfo('VNbMoTaW',  'bcc', 3.195)]

temperatures = [50, 300, 800, 1300, 1700, 2300, 3000, 4000, 5000]
for symbol_info in symbol_infos:
    for temperature in temperatures:
        atoms = symbol_info.create_bulk_atoms((3,3,3))
        dirname = f'{symbol_info.formula}/{symbol_info.lattice_type}/{temperature}K/relax'
        run_in=['potential nep.txt',
                f'velocity {temperature}',   
                'time_step 1',
                f'ensemble npt_mttk temp {temperature} {temperature} iso 0 0',
                'dump_thermo 10000',
                'dump_exyz 200000',
                'dump_restart 1000000',
                'run 1000000']
        Morph(atoms).gpumd(dirname=dirname, run_in=run_in, nep_path='../potentials/MoNbTaVW.txt')
