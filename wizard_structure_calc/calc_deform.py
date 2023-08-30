from wizard.atoms import SymbolInfo
from wizard.io import write_run, dump_xyz, run_gpumd
from ase.build import bulk
import os

def mkdir_deform(atoms):
    if not os.path.exists(f'deform'):
        os.makedirs(f'deform')
    original_directory = os.getcwd()
    os.chdir(f'deform')
    dump_xyz('model.xyz', atoms)
    write_run(['potential ../nep.txt', 'velocity 300', 'time_step 1',
               'ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000','run 1000000', 
               'ensemble npt_scr 300 300 100 0 0 0 100 100 100 1000',
               'deform 0.00001 1 0 0', 'dump_thermo 1000', 'dump_position 1000', 'run 1000000'])
    os.chdir(original_directory)

def main():
    lc = 3.1854
    atoms = bulk('W', 'bcc', lc, cubic = True) * (50, 25, 25)
    mkdir_deform(atoms)
    run_gpumd('deform')
        
if __name__ == "__main__":
    main()

