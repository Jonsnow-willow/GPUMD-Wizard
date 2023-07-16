# GPUMD-Wizard
Material structure processing software based on [ASE](https://wiki.fysik.dtu.dk/ase/index.html) (Atomic Simulation Environment).  It provides automation capabilities for calculating various properties of metals.

## Features
* Based on the ASE package, MetalProperties-Automator supports different calculators such as [PyNEP](https://github.com/bigd4/PyNEP), [calorine](https://calorine.materialsmodeling.org/installation.html#installation-via-pip), [DP](https://github.com/deepmodeling/deepmd-kit), and [LAMMPS](https://www.lammps.org/).
* Allows for automated batch calculations of metal properties.
* Enables batch processing of files in the XYZ format.
* Integrated with GPUMD for performing molecular dynamics simulations, such as irradiation damage.

## Installation


### Requirements


|  Package  | version |
|  ----  | ----  |
| [Python](https://www.python.org/) | >=     3.8 |
|[ase](https://wiki.fysik.dtu.dk/ase/index.html)|>=     3.18.0|
| [elastic](https://github.com/jochym/Elastic) | |
| [PyNEP](https://github.com/bigd4/PyNEP) | |


 ### From Source

```shell
$ git clone --recursive https://github.com/Jonsnow-willow/GPUMD-Wizard.git
```

Add `GPUMD-Wizard` to your [`PYTHONPATH`](https://wiki.fysik.dtu.dk/ase/install.html#envvar-PYTHONPATH) environment variable in your `~/.bashrc` file.

```shell
$ export PYTHONPATH=<path-to-GPUMD-Wizard-package>:$PYTHONPATH
```

## Usage
```python
from wizard.atoms import SymbolInfo, MaterialCalculator
from pynep.calculate import NEP
#from deepmd.calculator import DP
#from ase.calculators.lammpslib import LAMMPSlib


def main():
    # Create calculator object 
    calc = NEP('nep.txt')
    # calc = DP('dp.pb')
    # cmds = ["pair_style eam/alloy",
    #         "pair_coeff * * W.eam.alloy W]
    # calc = LAMMPSlib(lmpcmds=cmds, log_file='log.' + symbol_info.symbol, keep_alive=True)

    # Set properties-related parameters
    millers = [(1,1,0),(0,0,1),(1,1,1),(1,1,2)]
    sia_vectors = [(1/2,1/2,1/2),(1,0,0),(1,1,0)]
    nths = [1,2,3]

    # Generate bulk atoms and calculate properties
    symbol_info = SymbolInfo('W', 'bcc', 3.185)    
    atoms = symbol_info.create_bulk_atoms()
    material_calculator = MaterialCalculator(atoms, calc, symbol_info.symbol, symbol_info.structure)
    material_calculator.lattice_constant()
    material_calculator.elastic_constant()
    material_calculator.eos_curve()
    material_calculator.phonon_dispersion()
    material_calculator.formation_energy_vacancy()
    material_calculator.migration_energy_vacancy()
    for nth in nths:
        material_calculator.formation_energy_divacancies(nth)
    for miller in millers:
        material_calculator.formation_energy_surface(miller)
    material_calculator.stacking_fault(a = (-1,1,1), b = (1,1,0), distance = 3.185/2)
    material_calculator.stacking_fault(a = (1,1,-1), b = (1,1,2), distance = 3.185/2)
    material_calculator.pure_bcc_metal_screw_dipole_move()
    material_calculator.pure_bcc_metal_screw_one_move()
    for vector in sia_vectors:
        material_calculator.formation_energy_sia(vector)
    material_calculator.formation_energy_interstitial_atom('W',[0,0,1/2],'octahedral')
    material_calculator.formation_energy_interstitial_atom('W',[1/4,0,1/2],'tetrahedral')

if __name__ == "__main__":
    main()
```
