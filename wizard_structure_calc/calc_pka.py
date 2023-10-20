from ase.build import bulk
from wizard.io import write_run, run_gpumd, group_xyz, set_pka
import numpy as np
import os

# main_dir{potential.txt radiation0{generator_hea.py relax cascade}} 
def mkdir_relax(atoms, lc, duplicate):
    if os.path.exists('relax'):
        raise FileExistsError('Directory "relax" already exists')
    os.makedirs('relax')
    original_directory = os.getcwd()
    os.chdir('relax')
    group_xyz('model.xyz', atoms, [3 * lc, 3 * lc, 3 * lc], [(duplicate - 3) * lc, (duplicate - 3) * lc, (duplicate - 3) * lc])
    write_run(['potential ../../nep.txt', 'velocity 300', 'time_step 1', 
              'ensemble npt_scr 300 300 200 0 500 2000', 
              'dump_thermo 1000', 'dump_restart 30000', 'run 30000'])
    os.chdir(original_directory)

def mkdir_cascade(path, pka_energy, angle, index):
    if os.path.exists(path):
        raise FileExistsError('Directory already exists')
    os.makedirs(path)
    original_directory = os.getcwd()
    os.chdir(path)
    set_pka('../relax/restart.xyz', pka_energy, angle, index)
    write_run(['potential ../../nep.txt', 'velocity 300', 'time_step 0', 
              'ensemble nve','dump_exyz 1 1 1', 'run 1',
              'time_step 1 0.015', 'ensemble heat_nhc 300 200 0 0 1',
              'compute 0 500 10 temperature', 'dump_restart 10000', 
              'dump_exyz 1000 1 1','run 70000'])
    os.chdir(original_directory)

def main():
    lc = 3.1854
    duplicate = 30
    pka_energy = 5000
    angle = np.array([1, 3, 5])
    index = int(2 * (duplicate * duplicate * duplicate / 2 + duplicate * duplicate / 2 + duplicate / 2))
    
    for i in range(10): 
        os.makedirs(f'radiation{i}')
        os.chdir(f'radiation{i}')
        atoms = bulk('W', 'bcc', lc, cubic = True) * (duplicate, duplicate, duplicate)

        mkdir_relax(atoms, lc, duplicate)
        run_gpumd('relax')
    
        mkdir_cascade('cascade', pka_energy, angle, index)
        run_gpumd('cascade')
        os.chdir('../')

if __name__ == '__main__':
    main()

