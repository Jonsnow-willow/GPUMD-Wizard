from ase.optimize import BFGS, LBFGS, FIRE, GPMin
from ase.optimize.sciopt import SciPyFminBFGS
from ase.constraints import ExpCellFilter
from ase.units import GPa
from ase import Atoms

def relax(atoms: Atoms,
          fmax: float = 0.001,
          steps: int = 500,
          minimizer: str = 'bfgs',
          constant_cell: bool = False,
          constant_volume: bool = False,
          hydrostatic_strain: bool = False,
          scalar_pressure: float = 0.0,
          **kwargs) -> None:
    if constant_cell: 
        ucf = atoms
    else:
        ucf = ExpCellFilter(atoms, hydrostatic_strain=hydrostatic_strain, 
                            constant_volume=constant_volume, 
                            scalar_pressure=scalar_pressure * GPa) 
    kwargs['logfile'] = kwargs.get('logfile', None)
    if minimizer == 'bfgs':
        dyn = BFGS(ucf, **kwargs)
        dyn.run(fmax=fmax, steps=steps)
    elif minimizer == 'lbfgs':
        dyn = LBFGS(ucf, **kwargs)
        dyn.run(fmax=fmax, steps=steps)
    elif minimizer == 'bfgs-scipy':
        dyn = SciPyFminBFGS(ucf, **kwargs)
        dyn.run(fmax=fmax, steps=steps)
    elif minimizer == 'fire':
        dyn = FIRE(ucf, **kwargs)
        dyn.run(fmax=fmax, steps=steps)
    elif minimizer == 'gpmin':
        dyn = GPMin(ucf, **kwargs)
        dyn.run(fmax=fmax, steps=steps)
    else:
        raise ValueError(f'Unknown minimizer: {minimizer}')
    