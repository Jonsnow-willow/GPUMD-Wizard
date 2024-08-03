from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from wizard.io import dump_xyz
from wizard.atoms import Morph
from datetime import datetime

class MolecularDynamics:
    def __init__(self, atoms, calc = None):
        self.atoms = atoms
        self.calc = calc
        if calc is None:
            print("Using gpumd for molecular dynamics.")

    def report(self, steps):  
        atoms = self.atoms
        epot = atoms.get_potential_energy() / len(atoms)
        ekin = atoms.get_kinetic_energy() / len(atoms)
        print('Step: %d  Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV' % (steps, epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

    def dump_interval(self):
        dump_xyz('dump.xyz', self.atoms)

    def deform(self, A_per_step):
        cell = self.atoms.get_cell()
        cell[2, 2] += A_per_step
        self.atoms.set_cell(cell, scale_atoms = False)

    def NVE(self, temperature_K = 300, steps = 10000, dump_interval = 1000, timestep = 1):
        atoms = self.atoms
        if self.calc is None:
            dirname = f"gpumd_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            Morph(atoms).gpumd(dirname=dirname, 
                               run_in = ['potential nep.txt', f'velocity {temperature_K}', f'time_step {timestep}', 
                                         'ensemble nve', f'dump_exyz {dump_interval}', f'run {steps}'])
        else:
            atoms.calc = self.calc
            timestep = timestep * units.fs
            report_interval = steps // 10
            if 'velocities' not in atoms.info or atoms.info['velocities'] is None:
                MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
            dyn = VelocityVerlet(atoms, timestep)
            dyn.attach(lambda: self.report(dyn.nsteps), interval=report_interval)
            dyn.attach(self.dump_interval, interval=dump_interval)
            dyn.run(steps)

    def NVT(self, temperature_K = 300, steps = 10000, dump_interval = 1000, timestep = 1, A_per_step = None):
        atoms = self.atoms
        if self.calc is None:
            dirname = f"gpumd_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            Morph(atoms).gpumd(dirname=dirname, 
                               run_in = ['potential nep.txt', f'velocity {temperature_K}', f'time_step {timestep}', 
                                         f'ensemble nvt_ber {temperature_K} {temperature_K} 200', 
                                         f'dump_exyz {dump_interval}', f'run {steps}'])
        else:
            atoms.calc = self.calc
            timestep = timestep * units.fs
            report_interval = steps // 10
            if 'velocities' not in self.atoms.info or self.atoms.info['velocities'] is None:
                MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)
            dyn = NVTBerendsen(atoms, timestep=timestep, temperature_K=temperature_K, taut=100 * units.fs)
            dyn.attach(lambda: self.report(dyn.nsteps), interval=report_interval)
            dyn.attach(self.dump_interval, interval=dump_interval)
            if A_per_step is not None:
                dyn.attach(lambda: self.deform(A_per_step))
            dyn.run(steps)

    def NPT(self, temperature_K = 300, pressure = 0, steps = 10000, dump_interval = 1000, timestep = 1):
        atoms = self.atoms
        if self.calc is None:
            dirname = f"gpumd_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            Morph(atoms).gpumd(dirname=dirname, 
                               run_in = ['potential nep.txt', f'velocity {temperature_K}', f'time_step {timestep}', 
                                         f'ensemble npt_mttk temp {temperature_K} {temperature_K} iso {pressure} {pressure}', 
                                         f'dump_exyz {dump_interval}', f'run {steps}'])
        else:
            atoms.calc = self.calc
            timestep = timestep * units.fs
            report_interval = steps // 10
            if 'velocities' not in self.atoms.info or self.atoms.info['velocities'] is None:
                MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)
            dyn = NPTBerendsen(self.atoms, timestep=timestep, temperature_K=temperature_K,
                    taut=100 * units.fs, pressure=pressure,
                    taup=1000 * units.fs, compressibility_au=4.57e-5 / units.bar)
            dyn.attach(lambda: self.report(dyn.nsteps), interval=report_interval)
            dyn.attach(self.dump_interval, interval=dump_interval)
            dyn.run(steps)

    def MCMD(self, temperature_K = 300, pressure = 0, steps = 10000, dump_interval = 1000, timestep = 1):
        atoms = self.atoms
        if self.calc is None:
            dirname = f"gpumd_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            Morph(atoms).gpumd(dirname=dirname, 
                               run_in = ['potential nep.txt', f'velocity {temperature_K}', f'time_step {timestep}', 
                                         f'ensemble npt_mttk temp {temperature_K} {temperature_K} iso {pressure} {pressure}', 
                                         f'mc canonical 100 100 {temperature_K} {temperature_K}',
                                         f'dump_exyz {dump_interval}', f'run {steps}'])
        else:
            raise RuntimeError("MCMD is not implemented without gpumd.")

