from ase.build import bulk
from wizard.io import write_run, run_gpumd, dump_xyz, set_pka
from wizard.atoms import Morph
import numpy as np
import os
import shutil

# main_dir{potential.txt radiation0{generator_hea.py relax cascade}} 
def mkdir_relax(atoms):
    if not os.path.exists('relax'):
        os.makedirs('relax')
    original_directory = os.getcwd()
    os.chdir('relax')
    dump_xyz('model.xyz', atoms)
    write_run(['potential ../../nep.txt', 'velocity 300', 'time_step 1', 
              'ensemble npt_scr 300 300 200 0 500 2000', 'dump_thermo 1000', 
              'dump_restart 20000', 'dump_exyz 2000 1 1', 'run 20000'])
    os.chdir(original_directory)

def mkdir_cascade(path, pka_energy, angle, index):
    if not os.path.exists(path):
        os.makedirs(path)
    original_directory = os.getcwd()
    os.chdir(path)
    set_pka('../relax/restart.xyz', pka_energy, angle, index, False)
    write_run(['potential ../../nep.txt', 'velocity 300', 
              'time_step 1 0.015', 'ensemble nve',
              'dump_thermo 1000', 'dump_restart 30000', 
              'dump_exyz 3000 1 1','run 30000'])
    os.chdir(original_directory)

def main():
    lc = 3.185
    duplicate = 80
    atoms = bulk('W', 'bcc', lc, cubic = True) * (duplicate, duplicate, duplicate)

    pka_energy = 5000
    index = int(2 * (duplicate * duplicate * duplicate / 2 + duplicate * duplicate / 2 + duplicate / 2))
    theta = np.random.uniform(0, np.pi)  
    phi = np.random.uniform(0, 2 * np.pi)  
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    angle = np.array([x, y, z])
    
    if not os.path.exists('radiation_0'):
        os.makedirs('radiation_0')

    os.chdir('radiation_0')
    mkdir_relax(atoms)
    run_gpumd('relax')
    mkdir_cascade('cascade', pka_energy, angle, index)
    run_gpumd('cascade')
    os.chdir('../')

    for i in range(1,501):
        folder_name = 'radiation_' + str(i)
        copy_name = 'radiation_' + str(i - 1)

        os.makedirs(os.path.join(folder_name, 'relax'), exist_ok=True)
        shutil.copy(os.path.join(copy_name, 'cascade', 'restart.xyz'), os.path.join(folder_name, 'relax', 'model.xyz'))
        shutil.copy(os.path.join(copy_name, 'relax', 'run.in'), os.path.join(folder_name, 'relax', 'run.in'))
        
        os.chdir(folder_name)
        run_gpumd('relax')
        mkdir_cascade('cascade', pka_energy, angle, index)
        run_gpumd('cascade')
        os.chdir('../')

if __name__ == '__main__':
    main()

