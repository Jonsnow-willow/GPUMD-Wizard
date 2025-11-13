from wizard.io import read_xyz
from ase.calculators.vasp import Vasp
import shutil
import os

frames = read_xyz('train.xyz')

for i, atoms in enumerate(frames):
    dir_name = f"{i}"  
    
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    calc = Vasp(directory=dir_name,
                encut=500,
                ediff=1e-5,
                kspacing=0.2,
                sigma=0.1,
                ismear=1,
                gga='PE',
                prec='Accurate',
                lasph=True,
                nelmin=4,
                nelm=200,
                lcharge=False,
                lwave=False)

    calc.write_input(atoms)