from __future__ import annotations

import time
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .ensembles import NVE, NVTBerendsen, NPTBerendsen, NVTLangevin
from .mc import MonteCarlo, Canonical, SGC, VCSGC, GC
from ..utils.io import dump_xyz


class MolecularDynamics:
    """Core MD driver.
    Integrators (ensemble arg):
      - nve: Velocity Verlet
      - nvt_ber: Berendsen thermostat
      - nvt_lan: Langevin (Bussi-Parrinello)
      - npt_ber: Berendsen thermo+barostat
    Steps atoms, logs thermo, optional dumps.
    """

    def __init__(self, atoms: Atoms, calc=None, temperature_K: float | None = None):
        self.atoms = atoms
        if calc is None:
            raise ValueError("A calculator must be provided for NumPy-based MD.")
        self.atoms.calc = calc
        if temperature_K is not None:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K)
            Stationary(self.atoms)
            print(f"INFO: Initialized velocities using Maxwell-Boltzmann distribution at {temperature_K} K.")
        elif self.atoms.get_velocities() is not None:
            print("INFO: Using initial velocities from atoms.")
        else:
            raise ValueError("No velocities found; provide temperature_K to initialize.")

    def _report(self, steps: int, thermo: dict[str, float | None]):
        natoms = len(self.atoms)
        epot = thermo["potential"] / natoms
        ekin = thermo["kinetic"] / natoms
        temperature = thermo["temperature"]
        etot = thermo["total"] / natoms
        pressure = "N/A" if (pressure := thermo.get("pressure")) is None else f"{pressure:.3f}"
        print("Step: %d  Energy per atom: \"Epot = %.3f eV  Ekin = %.3f eV (T=%3.0fK)  Etot = %.3f eV\"  P=%s GPa"
              % (steps, epot, ekin, temperature, etot, pressure))

    def _dump_interval(self):
        dump_xyz("dump.xyz", self.atoms)

    _ENSEMBLE_BUILDERS = {
        "nve": "_build_nve",
        "nvt_ber": "_build_nvt_ber",
        "nvt_lan": "_build_nvt_lan",
        "npt_ber": "_build_npt_ber",
    }

    def _initialize_integrator(self, ensemble: str, *, timestep_fs: float, **ensemble_kwargs):
        builder_name = self._ENSEMBLE_BUILDERS.get(ensemble.lower())
        if builder_name is None:
            raise ValueError(f"Unknown ensemble '{ensemble}'.")
        builder = getattr(self, builder_name)
        return builder(timestep_fs=timestep_fs, **ensemble_kwargs)

    def _build_nve(self, *, timestep_fs: float, **_):
        return NVE(self.atoms, timestep_fs)

    def _build_nvt_ber(self, *, timestep_fs: float, temperature_K: float, tau_T: float, **_):
        return NVTBerendsen(self.atoms, timestep_fs, temperature_K, tau_T)

    def _build_nvt_lan(self, *, timestep_fs: float, temperature_K: float, tau_T: float, **_):
        return NVTLangevin(self.atoms, timestep_fs, temperature_K, tau_T)

    def _build_npt_ber(self,
        *,
        timestep_fs: float,
        temperature_K: float,
        tau_T: float,
        pressure_GPa: float,
        tau_P: float,
        C_hydro: float,
        **_,
    ):
        return NPTBerendsen(
            self.atoms,
            timestep_fs,
            temperature_K,
            tau_T,
            pressure_GPa,
            tau_P,
            C_hydro,
        )
    
    def run(
        self,
        ensemble: str,
        *,
        timestep_fs: float = 1.0,
        steps: int = 5000,
        dump_interval: int | None = None,
        report_interval: int | None = None,
        **ensemble_kwargs,
    ):
        atoms = self.atoms
        integrator = self._initialize_integrator(
            ensemble,
            timestep_fs=timestep_fs,
            **ensemble_kwargs,
        )
        start_time = time.perf_counter()
        forces = atoms.get_forces(md=True)
        self._report(steps=0, thermo=integrator.thermo)
        for step in range(1, steps + 1):
            integrator.compute1(forces)
            forces = atoms.get_forces(md=True)
            integrator.compute2(forces)
            integrator.update_thermo()
            if report_interval and report_interval > 0 and step % report_interval == 0:
                self._report(step, thermo=integrator.thermo)
            if dump_interval and dump_interval > 0 and step % dump_interval == 0:
                self._dump_interval()
        time_used = time.perf_counter() - start_time
        natoms = len(atoms)
        speed = natoms * steps / time_used if steps > 0 else 0.0
        print(f"Time used for this run = {time_used:.5f} second.")
        print(f"Speed of this run = {speed:.0f} atom*step/second.")


class MCMD(MolecularDynamics):
    """Hybrid MD + MC driver.
    MC schemes (mc_scheme arg):
      - swap: identity swap
      - canonical: Boltzmann swap at T
      - sgc: semi-grand-canonical (μ bias)
      - vcsgc: variance-constrained SGC
      - gc: grand-canonical insertion/deletion
      - lgc: lattice-restricted GC
    Interleaves chosen MD integrator with MC moves.
    """

    _MC_Builders = {
        "swap": "_build_swap", 
        "canonical": "_build_canonical",
        "sgc": "_build_sgc",
        "vcsgc": "_build_vcsgc",
        "gc" : "_build_gc",
        "lgc": "_build_lgc",
    }

    def _initialize_mc(self, mc_scheme: str, *, md_steps: int, mc_trials: int, **ensemble_kwargs):
        builder_name = self._MC_Builders.get(mc_scheme.lower())
        if builder_name is None:
            raise ValueError(f"Unknown MC ensemble '{mc_scheme}'.")
        builder = getattr(self, builder_name)
        return builder(md_steps=md_steps, mc_trials=mc_trials, **ensemble_kwargs)

    def _build_swap(self, *, md_steps: int, mc_trials: int, **_):
        return MonteCarlo(self.atoms, md_steps, mc_trials)    

    def _build_canonical(self, *, md_steps: int, mc_trials: int, temperature_K: float, **_):
        return Canonical(self.atoms, md_steps, mc_trials, temperature_K)
    
    def _build_sgc(self, *, md_steps: int, mc_trials: int, temperature_K: float, mus: dict[str, float], **_):
        return SGC(self.atoms, md_steps, mc_trials, temperature_K, mus)
    
    def _build_vcsgc(self, *, md_steps: int, mc_trials: int, temperature_K: float, mus: dict[str, float], kappa: float, **_):
        return VCSGC(self.atoms, md_steps, mc_trials, temperature_K, mus, kappa)
    
    def _build_gc(self, *, md_steps: int, mc_trials: int, temperature_K: float, mus: dict[str, float], **_):
        return GC(self.atoms, md_steps, mc_trials, temperature_K, mus)

    def run(
        self,
        ensemble: str,
        mc_scheme: str,
        *,
        timestep_fs: float = 1.0,
        steps: int = 5000,
        md_steps: int = 100,
        mc_trials: int = 100,
        dump_interval: int | None = None,
        report_interval: int | None = None,
        **ensemble_kwargs,
    ):
        atoms = self.atoms
        integrator = self._initialize_integrator(
            ensemble,
            timestep_fs=timestep_fs,
            **ensemble_kwargs,
        )
        mc = self._initialize_mc(
            mc_scheme,
            md_steps=md_steps,
            mc_trials=mc_trials,
            **ensemble_kwargs,
        )
        start_time = time.perf_counter()
        forces = atoms.get_forces(md=True)
        self._report(0, thermo=integrator.thermo)
        for step in range(1, steps + 1):
            integrator.compute1(forces)
            forces = atoms.get_forces(md=True)
            integrator.compute2(forces)
            mc.compute(step)
            if step % md_steps == 0:
                forces = atoms.get_forces(md=True)
                integrator.update_masses()
            integrator.update_thermo()
            if report_interval and report_interval > 0 and step % report_interval == 0:
                self._report(step, thermo=integrator.thermo)
            if dump_interval and dump_interval > 0 and step % dump_interval == 0:
                self._dump_interval()
        time_used = time.perf_counter() - start_time
        natoms = len(atoms)
        speed = natoms * steps / time_used if steps > 0 else 0.0
        print(f"Time used for this run = {time_used:.5f} second.")
        print(f"Speed of this run = {speed:.0f} atom*step/second.")