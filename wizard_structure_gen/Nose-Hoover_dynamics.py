from pynep.calculate import NEP
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from wizard.io import dump_xyz

atoms = bulk("W", "bcc", a=3.1652, cubic=True) * (3, 3, 3)
atoms.calc = NEP('nep.txt')

# Set the Maxwell-Boltzmann velocity distribution for the atoms
temperature_K = 1200 * units.kB # in Kelvin
MaxwellBoltzmannDistribution(atoms, temperature_K)

# Initialize the NPT thermostat
timestep = 1.0 * units.fs  # 1 fs timestep
external_pressure = 0.0  # External pressure in bar
ttime = 100.0 * units.fs # Adjust the ttime as needed
pfactor = 500 * units.GPa

npt = NPT(atoms, timestep, temperature_K, external_pressure * units.bar, ttime, pfactor)

# Run the simulation
nsteps = 10
for step in range(nsteps):
    npt.run(1000)
    dump_xyz('example/molecular_dynamics/Nose-Hoover.xyz', atoms)