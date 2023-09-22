from ase.build import bulk
from wizard.io import mkdir_relax, run_gpumd
from wizard.atoms import Morph
import os

def main():
    lc = 3.1854
    duplicate = 200
    atoms = bulk('W', 'bcc', lc, cubic = True) * (duplicate, duplicate, duplicate)
    Morph(atoms).prop_element_set(['V','Nb','Mo','Ta','W'])
    mkdir_relax(atoms)
    run_gpumd('relax')

if __name__ == '__main__':
    main()

