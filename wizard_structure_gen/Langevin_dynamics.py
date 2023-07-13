from pynep.calculate import NEP
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from wizard.io import dump_xyz

atoms = bulk("W", "bcc", a=3.1652, cubic=True) * (3, 3, 3)
atoms.calc = NEP('nep.txt')

# Set the Maxwell-Boltzmann velocity distribution for the atoms
temperature = 300  # in Kelvin
MaxwellBoltzmannDistribution(atoms, temperature * units.kB)

# Initialize the NVT Berendsen thermostat
timestep = 1.0 * units.fs  # 1 fs timestep
friction = 0.002  # Friction factor (gamma)

langevin = Langevin(atoms, timestep, temperature * units.kB, friction)

# Run the simulation
nsteps = 10
for step in range(nsteps):
    langevin.run(1000)
    dump_xyz('example/molecular_dynamics/Langevin.xyz', atoms)
