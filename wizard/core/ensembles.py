from __future__ import annotations

import numpy as np
from ase import units
from ase.atoms import Atoms

class VelocityVerlet:
    """Integrator implementing the velocity-Verlet scheme."""
    
    def __init__(self, atoms: Atoms, timestep_fs: float):
        self.atoms = atoms
        if timestep_fs is None or timestep_fs <= 0.0:
            raise ValueError("timestep_fs must be a positive value.")
        self.dt = timestep_fs * units.fs
        self.masses = atoms.get_masses()[:, None]
        self.thermo: dict[str, float | None] = {}
        self.update_thermo()

    def _half_kick(self, forces: np.ndarray):
        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * forces
        self.atoms.set_momenta(p, apply_constraint=False)

    def _drift(self):
        p = self.atoms.get_momenta()
        positions = self.atoms.get_positions()
        self.atoms.set_positions(positions + self.dt * p / self.masses)

    def _after_velocities_updated(self):
        """Hook executed after velocity update."""
        pass

    def compute1(self, forces: np.ndarray):
        """First half-step: kick and drift before forces are re-evaluated."""
        self._half_kick(forces)
        self._drift()

    def compute2(self, forces: np.ndarray):
        """Second half-step: kick again, then allow subclasses to act."""
        self._half_kick(forces)
        self._after_velocities_updated()

    def update_thermo(self):
        atoms = self.atoms
        thermo = {
            "kinetic": atoms.get_kinetic_energy(),
            "potential": atoms.get_potential_energy(),
            "temperature": atoms.get_temperature(),
            "pressure": None,
        }
        try:
            stress = atoms.get_stress(voigt=True)
        except Exception:
            pass
        else:
            thermo["pressure"] = -np.mean(stress[:3]) / units.GPa
        thermo["total"] = thermo["kinetic"] + thermo["potential"]
        self.thermo = thermo

    def update_masses(self):
        self.masses = self.atoms.get_masses()[:, None]


class NVE(VelocityVerlet):
    """Plain NVE integrator."""

    pass


class NVTBerendsen(VelocityVerlet):
    """Berendsen thermostat applied after the velocity update."""

    def __init__(self, atoms: Atoms, timestep_fs: float, temperature_K: float, tau_T: float):
        super().__init__(atoms, timestep_fs)
        if temperature_K is None or temperature_K <= 0.0:
            raise ValueError("temperature_K must be provided for Berendsen thermostat.")
        self.target_temperature = temperature_K
        if tau_T is None or tau_T <= 0.0:
            raise ValueError("tau_T must be a positive value for Berendsen thermostat.")
        self.tau_T = tau_T * units.fs

    def _after_velocities_updated(self):
        self._apply_thermostat()

    def _apply_thermostat(self):
        current_T = self.atoms.get_temperature()
        if current_T <= 1e-12:
            return
        factor_sq = 1.0 + self.dt / self.tau_T * (self.target_temperature / current_T - 1.0)
        factor_sq = max(factor_sq, 0.0)
        scale = np.sqrt(factor_sq)
        p = self.atoms.get_momenta()
        self.atoms.set_momenta(scale * p, apply_constraint=False)


class NPTBerendsen(NVTBerendsen):
    """Berendsen thermostat coupled with an isotropic Berendsen barostat."""

    def __init__(
        self,
        atoms: Atoms,
        timestep_fs: float,
        temperature_K: float,
        tau_T: float,
        pressure_GPa: float,
        tau_P: float,
        C_hydro: float,
    ):
        super().__init__(atoms, timestep_fs, temperature_K, tau_T)
        if pressure_GPa is None:
            raise ValueError("pressure_GPa must be provided for Berendsen barostat.")
        self.target_pressure = pressure_GPa
        if tau_P is None or tau_P <= 0.0:
            raise ValueError("tau_P must be a positive value for Berendsen barostat.")
        self.tau_P = tau_P * units.fs
        if C_hydro is None or C_hydro <= 0.0:
            raise ValueError("C_hydro must be positive.")
        self.C_hydro = C_hydro

    def _after_velocities_updated(self):
        self._apply_thermostat()
        self._apply_barostat()

    def _apply_barostat(self):
        try:
            stress = self.atoms.get_stress(voigt=True)
        except Exception:
            raise RuntimeError("Unable to get stress for barostat.")
        current_P = -np.mean(stress[:3]) / units.GPa
        scale = 1.0 - (self.dt / (self.tau_P * 3 * self.C_hydro)) * (self.target_pressure - current_P)
        scale = max(scale, 1e-8)
        cell = self.atoms.get_cell()
        self.atoms.set_cell(cell * scale, scale_atoms=True)


class NVTLangevin(VelocityVerlet):
    """Bussi-Parrinello Langevin thermostat."""

    def __init__(self, atoms: Atoms, timestep_fs: float, temperature_K: float, tau_T: float):
        super().__init__(atoms, timestep_fs)
        if timestep_fs is None or timestep_fs <= 0.0:
            raise ValueError("temperature_K must be provided for Langevin thermostat.")
        if tau_T is None or tau_T <= 0.0:
            raise ValueError("tau_T must be a positive value for Langevin thermostat.")
        self.c1 = np.exp(-0.5 * timestep_fs / tau_T)
        self.c2 = np.sqrt((1.0 - self.c1 * self.c1) * units.kB * temperature_K)
        self._rng = np.random.default_rng()

    def _after_velocities_updated(self):
        self._apply_langevin()

    def _apply_langevin(self):
        masses = self.masses
        momenta = self.atoms.get_momenta()
        if momenta is None:
            raise RuntimeError("Velocities must be initialized before applying Langevin thermostat.")
        noise = self._rng.normal(size=momenta.shape)
        momenta = self.c1 * momenta + self.c2 * np.sqrt(masses) * noise
        
        total_mass = float(np.sum(masses))
        total_momentum = np.sum(momenta, axis=0)
        momenta -= masses * (total_momentum / total_mass)
        self.atoms.set_momenta(momenta, apply_constraint=False)

    def compute1(self, forces: np.ndarray):
        self._apply_langevin()
        super().compute1(forces)
