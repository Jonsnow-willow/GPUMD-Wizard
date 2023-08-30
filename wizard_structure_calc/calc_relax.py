from ase.build import bulk
from wizard.io import write_run, dump_xyz, run_gpumd
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
              'ensemble npt_scr 300 300 200 0 500 2000', 
              'dump_thermo 1000', 'dump_restart 30000', 'run 30000'])
    os.chdir(original_directory)

def main():
    lc = 3.1854
    duplicate = 200
    atoms = bulk('W', 'bcc', lc, cubic = True) * (duplicate, duplicate, duplicate)
    Morph(atoms).prop_element_set(['V','Nb','Mo','Ta','W'])
    mkdir_relax(atoms)
    run_gpumd('relax')

if __name__ == '__main__':
    main()

