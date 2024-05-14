from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.io.trajectory import Trajectory

class MolecularDynamics:
    def __init__(self, atoms, calc, timestep = units.fs, dump_interval = 100, report_interval = 100, A_per_step = None):
        self.atoms = atoms
        self.calc = calc
        self.timestep = timestep
        self.dump_interval = dump_interval
        self.report_interval = report_interval
        self.nsteps = 0
        self.A_per_step = A_per_step

    def report(self):  
        atoms = self.atoms
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        print('Step: %d Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
            'Etot = %.3feV' % (self.nsteps, epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
        self.nsteps += self.report_interval

    def deform(self):
        atoms = self.atoms
        cell = atoms.get_cell()
        cell[2, 2] += self.A_per_step
        atoms.set_cell(cell, scale_atoms=True)

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
        dyn = NVTBerendsen(self.atoms, timestep=self.timestep, temperature_K=temperature_K, taut=100 * units.fs)
        dyn.attach(self.report, interval=self.report_interval)
        dyn.attach(traj.write, interval=self.dump_interval)
        if self.A_per_step is not None:
            dyn.attach(self.deform)
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