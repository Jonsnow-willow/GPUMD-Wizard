from wizard.atoms import SymbolInfo
from wizard.io import write_run, group_xyz, run_gpumd
import os

def mkdir_relax(atoms, height, symbol, Tm):
    if not os.path.exists(f'relax_{symbol}'):
        os.makedirs(f'relax_{symbol}')
    original_directory = os.getcwd()
    Ti = Tm * 0.5
    Tcut = Tm - Ti
    os.chdir(f'relax_{symbol}')
    group_xyz('model.xyz', atoms, [-10, -10, height], [10000000, 10000000, height])
    write_run(['potential ../nep.txt', f'velocity {Ti}', 'time_step 1', 
              f'ensemble npt_ber {Ti} {Ti} 200 0 0 0 0 0 0 500 500 500 500 500 500 2000', 
              'dump_exyz 10000 0 0', 'run 100000',
              f'ensemble heat_lan {Tm} 200 {Tcut} 0 1',
              'dump_exyz 10000 0 0','run 100000',
              f'ensemble npt_ber {Tm} {Tm} 1e20 0 0 0 0 0 0 500 500 500 500 500 500 2000', 
              'dump_thermo 100', 'dump_exyz 10000 0 0', 'run 100000'])
    os.chdir(original_directory)

def main():
    symbol_infos = [
    SymbolInfo('Ag', 'fcc', 4.146),
    SymbolInfo('Al', 'fcc', 4.042),
    SymbolInfo('Au', 'fcc', 4.159),
    SymbolInfo('Cu', 'fcc', 3.631),
    SymbolInfo('Ni', 'fcc', 3.509),
    SymbolInfo('Pb', 'fcc', 5.038),
    SymbolInfo('Pd', 'fcc', 3.939),
    SymbolInfo('Pt', 'fcc', 3.967),
    SymbolInfo('Cr', 'bcc', 2.845),
    SymbolInfo('Fe', 'bcc', 2.759),
    SymbolInfo('Mo', 'bcc', 3.164),
    SymbolInfo('Ta', 'bcc', 3.319),
    SymbolInfo('V', 'bcc', 2.997),
    SymbolInfo('W', 'bcc', 3.185),
    SymbolInfo('Co', 'hcp', 2.256, 6.180),
    SymbolInfo('Mg', 'hcp', 3.195, 5.186),
    SymbolInfo('Ti', 'hcp', 2.931, 4.651),
    SymbolInfo('Zr', 'hcp', 3.240, 5.157)
    ]
    T = [955, 839, 829, 1205, 1523, 591, 1458, 1475, 2195, 1712, 2478, 2643, 
         1991, 3229, 1741, 814, 1572, 1740]
    for symbol_info, t in zip(symbol_infos, T):
        atoms = symbol_info.create_bulk_atoms() * (20, 20, 20)
        symbol = symbol_info.symbol
        if symbol_info.structure == 'hcp':
            height = 10 * symbol_info.lattice_constant[1]
        else:
            height = 10 * symbol_info.lattice_constant[0]
        mkdir_relax(atoms, height, symbol, t)
        run_gpumd(f'relax_{symbol}')
        
if __name__ == "__main__":
    main()

