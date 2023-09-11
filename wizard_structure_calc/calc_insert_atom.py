from ase.build import bulk
from wizard.io import write_run, dump_xyz, run_gpumd, read_xyz
from wizard.atoms import Morph
import os

def mkdir_relax(atoms):
    if os.path.exists('relax'):
        raise FileExistsError('Directory "relax" already exists')
    os.makedirs('relax')
    original_directory = os.getcwd()
    os.chdir('relax')
    dump_xyz('model.xyz', atoms)
    write_run(['potential ../nep.txt', 'velocity 300', 'time_step 1', 
               'ensemble npt_scr 300 300 200 0 500 2000', 'dump_thermo 100', 
               'dump_position 100','dump_restart 1000', 'run 30000'])
    os.chdir(original_directory)

def mkdir_fks(atoms):
    if os.path.exists('fks'):
        raise FileExistsError('Directory "relax" already exists')
    os.makedirs('fks')
    original_directory = os.getcwd()
    os.chdir('fks')
    dump_xyz('model.xyz', atoms)
    write_run(['potential ../nep.txt', 'velocity 300', 'time_step 1', 
               'ensemble npt_scr 300 300 200 0 500 2000', 'dump_thermo 100', 
               'dump_position 1000','dump_restart 1000', 'run 1000000'])
    os.chdir(original_directory)

def main():
    lc = 3.1854
    duplicate = 100
    atoms = bulk('W', 'bcc', lc, cubic = True) * (duplicate, duplicate, duplicate)
    Morph(atoms).prop_element_set(['V','Nb','Mo','Ta','W'])
    mkdir_relax(atoms)
    run_gpumd('relax')
    atoms_sias = read_xyz('relax/restart.xyz')
    Morph(atoms_sias).create_fks(10000)
    mkdir_fks(atoms_sias)
    run_gpumd('fks')

if __name__ == '__main__':
    main()

