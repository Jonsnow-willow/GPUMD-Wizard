# GPUMD-Wizard
Material structure processing software based on ASE (Atomic Simulation Environment).  It provides automation capabilities for calculating various properties of metals.

## Features
* Based on the ASE package, MetalProperties-Automator supports different calculators such as pynep, DP, and LAMMPS.
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
