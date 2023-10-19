from ase.build import bulk
from wizard.io import mkdir_relax, run_gpumd
from wizard.atoms import Morph
import os

def main():
    lc = 3.1854
    duplicate = 200
    atoms = bulk('W', 'bcc', lc, cubic = True) * (duplicate, duplicate, duplicate)
    Morph(atoms).prop_element_set(['V','Nb','Mo','Ta','W'])
    mkdir_relax(atoms, ['potential ../nep.txt', 'velocity 2000', 'time_step 1', 
                        'ensemble npt_scr 2000 2000 200 0 500 2000', 
                        'dump_thermo 1000', 'dump_xyz 10000', 'dump_restart 10000', 'run 100000',
                        'ensemble npt_scr 2000 5000 200 0 500 2000',
                        'dump_thermo 1000', 'dump_xyz 100000', 'dump_restart 10000', 'run 10000000',
                        'ensemble npt_scr 4500 4500 200 0 500 2000',
                        'dump_thermo 1000', 'dump_xyz 10000', 'dump_restart 10000', 'run 100000',
                        'ensemble npt_scr 4500 1500 200 0 500 2000',
                        'dump_thermo 1000', 'dump_xyz 100000', 'dump_restart 10000', 'run 10000000',
                        ])
    run_gpumd('relax')

if __name__ == '__main__':
    main()

