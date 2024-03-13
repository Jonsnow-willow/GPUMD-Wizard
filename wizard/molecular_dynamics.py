from ase import units
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.io.trajectory import Trajectory

class MolecularDynamics:
    def __init__(self, atoms, calc, timestep = units.fs, dump_interval = 100, report_interval = 100):
        self.atoms = atoms
        self.calc = calc
        self.timestep = timestep
        self.dump_interval = dump_interval
        self.report_interval = report_interval
        self.nsteps = 0

    def report(self):  
        atoms = self.atoms
        self.nsteps += self.report_interval
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
            'Etot = %.3feV' % (self.nsteps, epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

    def NVE(self, temperature_K = 300, steps = 10000):
        atoms = self.atoms
        atoms.calc = self.calc
        traj = Trajectory('NVE.traj', 'w', atoms)
        if 'velocities' not in self.atoms.info or self.atoms.info['velocities'] is None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)
        dyn = VelocityVerlet(self.atoms, self.timestep)
        dyn.attach(self.report, interval=self.report_interval)
        dyn.attach(traj.write, interval=self.dump_interval)
        dyn.run(steps)

    def NVT(self, temperature_K = 300, steps = 10000):
        atoms = self.atoms
        atoms.calc = self.calc
        traj = Trajectory('NVT.traj', 'w', atoms)
        if 'velocities' not in self.atoms.info or self.atoms.info['velocities'] is None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)
        dyn = Langevin(self.atoms, self.timestep, temperature_K, 0.02)
        dyn.attach(self.report, interval=self.report_interval)
        dyn.attach(traj.write, interval=self.dump_interval)
        dyn.run(steps)

    def NPT(self, temperature_K = 300, pressure = 0, steps = 10000):
        atoms = self.atoms
        atoms.calc = self.calc
        traj = Trajectory('NPT.traj', 'w', atoms)
        if 'velocities' not in self.atoms.info or self.atoms.info['velocities'] is None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)
        dyn = NPTBerendsen(self.atoms, timestep=self.timestep, temperature_K=temperature_K,
                taut=100 * units.fs, pressure=pressure,
                taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)
        dyn.attach(self.report, interval=self.report_interval)
        dyn.attach(traj.write, interval=self.dump_interval)
        dyn.run(steps)

