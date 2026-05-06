from __future__ import annotations

import numpy as np
from ase import Atom, units
from ase.atoms import Atoms
from ase.data import chemical_symbols
from ase.geometry import geometry

K_B = units.kB
VALID_SYMBOLS = {s for s in chemical_symbols if s}

class MonteCarlo:
    """Base class for Monte Carlo swap moves."""

    def __init__(self, atoms: Atoms, md_steps: int, mc_trials: int):
        self.atoms = atoms
        if md_steps <= 0:
            raise ValueError("md_steps must be a positive integer.")
        self.md_steps = md_steps
        if mc_trials <= 0:
            raise ValueError("mc_trials must be a positive integer.")  
        self.mc_trials = mc_trials
        self._energy = float(self.atoms.get_potential_energy())
        self._indices = np.arange(len(atoms))
        self._rng = np.random.default_rng()

    @staticmethod
    def _coerce_symbol(raw) -> str:
        if hasattr(raw, "item"):
            try:
                raw = raw.item()
            except Exception:
                pass
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        symbol = str(raw).strip()
        if symbol.startswith("b'") and symbol.endswith("'") and len(symbol) >= 3:
            symbol = symbol[2:-1]
        if symbol not in VALID_SYMBOLS:
            raise ValueError(f"Invalid chemical symbol sampled in MC move: '{symbol}'")
        return symbol

    def _update_result(self, attempts: int, accepted: int, energy: float):
        ratio = accepted / attempts if attempts else 0.0
        dE_tot = energy - self._energy
        msg = (
            f"MC Attempts: {attempts}, Accepted: {accepted}, "
            f"Acceptance Ratio: {ratio:.2f}, "
            f"Energy Change: {dE_tot:.6f} eV."
        )
        with open("mcmd.out", "a", encoding="utf-8") as handle:
            handle.write(msg + "\n")

    def _accept(self, dE: float) -> bool:
        if dE <= 0.0:
            return True
        return False

    def compute(self, step: int):
        if step % self.md_steps != 0:
            return
        
        energy = self._energy
        attempts = self.mc_trials
        accepted = 0
        for _ in range(attempts):

            i, j = self._rng.choice(self._indices, size=2, replace=False)
            symbol_i = self.atoms[i].symbol
            symbol_j = self.atoms[j].symbol
            if symbol_i == symbol_j:
                continue

            self.atoms[i].symbol = symbol_j
            self.atoms[j].symbol = symbol_i

            new_energy = float(self.atoms.get_potential_energy())
            delta_e = new_energy - energy
            if self._accept(delta_e):
                accepted += 1
                energy = new_energy
            else:
                self.atoms[i].symbol = symbol_i
                self.atoms[j].symbol = symbol_j

        self._update_result(attempts, accepted, energy)


class Canonical(MonteCarlo):
    """Canonical MC swap moves that exchange atom identities within the supercell."""

    def __init__(self, atoms: Atoms, md_steps: int, mc_trials: int, temperature_K: float):
        super().__init__(atoms, md_steps, mc_trials)
        if temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive.")
        self.temperature_K = temperature_K

    def _accept(self, dE: float) -> bool:
        if dE <= 0.0:
            return True
        probability = np.exp(-dE / (K_B * self.temperature_K))
        return self._rng.random() < probability


class SGC(Canonical):
    """Semi-grand canonical MC moves including chemical potential bias."""

    def __init__(
        self,
        atoms: Atoms,
        md_steps: int,
        mc_trials: int,
        temperature_K: float,
        mus: dict[str, float]
    ):
        super().__init__(atoms, md_steps, mc_trials, temperature_K)
        self.mus = mus
        self.species = np.array(list(mus.keys()), dtype=object)
        self._update_indices()
        self._counts = {s: atoms.get_chemical_symbols().count(s) for s in self.species}
        self._count_sum = sum(self._counts.values())

    def _update_result(self, attempts: int, accepted: int, energy: float):
        ratio = accepted / attempts if attempts else 0.0
        dE_tot = energy - self._energy
        self._energy = energy
        num_atoms = len(self.atoms)
        symbols, counts = np.unique(self.atoms.get_chemical_symbols(), return_counts=True)
        composition =  {str(symbol): float(count) / num_atoms for symbol, count in zip(symbols, counts)}

        msg = (
            f"MC Attempts: {attempts}, Accepted: {accepted}, "
            f"Acceptance Ratio: {ratio:.2f}, "
            f"Energy Change: {dE_tot:.6f} eV, "
            f"Species Counts: {self._counts}, "
            f"Composition: {composition}."
        )
        with open("mcmd.out", "a", encoding="utf-8") as handle:
            handle.write(msg + "\n")
    
    def _update_indices(self):
        if len(self.species) < 2:
            raise ValueError("At least two species must be provided for SGC sampling.")
        symbols = np.array(self.atoms.get_chemical_symbols())
        self._indices = np.flatnonzero(np.isin(symbols, self.species))
    
    def _bias(self, old_species: str, new_species: str) -> float:
        return self.mus[new_species] - self.mus[old_species]
    
    def compute(self, step: int):
        if step % self.md_steps != 0:
            return
        
        energy = self._energy
        attempts = self.mc_trials
        accepted = 0
        for _ in range(attempts):

            i = self._rng.choice(self._indices)
            symbol_i = self.atoms[i].symbol
            symbol_j = self._coerce_symbol(self._rng.choice(self.species[self.species != symbol_i]))

            self.atoms[i].symbol = symbol_j
            new_energy = float(self.atoms.get_potential_energy())
            delta_e = new_energy - energy
            delta_mu = self._bias(symbol_i, symbol_j)
            delta_tot = delta_e + delta_mu

            if self._accept(delta_tot):
                accepted += 1
                energy = new_energy
                self._counts[symbol_i] -= 1
                self._counts[symbol_j] += 1
            else:
                self.atoms[i].symbol = symbol_i

        self._update_result(attempts, accepted, energy)


class VCSGC(SGC):
    """Variance-constrained semi-grand canonical MC swap moves that exchange atom identities within the supercell."""

    def __init__(
        self,
        atoms: Atoms,
        md_steps: int,
        mc_trials: int,
        temperature_K: float,
        mus: dict[str, float],
        kappa: float
    ):
        super().__init__(atoms, md_steps, mc_trials, temperature_K, mus)
        if kappa <= 0.0:
            raise ValueError("kappa must be positive for VCSGC sampling.")
        self.kappa = float(kappa)
        
    def _bias(self, old_species: str, new_species: str) -> float:
        delta_mu = super()._bias(old_species, new_species)
        count_diff = self._counts.get(new_species) - self._counts.get(old_species)
        constraint = (
            self.kappa
            * K_B
            * self.temperature_K
            / self._count_sum
            * (self._count_sum * delta_mu + 2.0 * count_diff + 1.0)
        )
        return delta_mu + constraint
    

class GC(SGC):
    """Grand canonical MC moves including chemical potential bias."""

    def _update_indices(self):
        if len(self.species) < 1:
            raise ValueError("At least one species must be provided for GC sampling.")
        symbols = np.array(self.atoms.get_chemical_symbols())
        self._indices = np.flatnonzero(np.isin(symbols, self.species))

    def _random_position(self) -> np.ndarray:
        min_distance = 1.2
        max_tries = 100
        cell = np.asarray(self.atoms.get_cell())
        positions = self.atoms.get_positions()
        has_cell = np.linalg.norm(cell) > 0.0
        pbc = self.atoms.get_pbc() if has_cell else None

        for _ in range(max_tries):
            if has_cell:
                pos = self._rng.random(3) @ cell
                _, dists = geometry.get_distances(
                    pos[None, :],
                    positions,
                    cell=cell,
                    pbc=pbc,
                )
                if dists.min() > min_distance:
                    return pos
            else:
                mins = positions.min(axis=0)
                maxs = positions.max(axis=0)
                pos = mins if np.allclose(mins, maxs) else self._rng.uniform(mins, maxs)
                if np.linalg.norm(positions - pos, axis=1).min() > min_distance:
                    return pos

        raise RuntimeError("Failed to sample insertion position beyond min_distance.")
    
    def _move(self):
        if self._rng.random() < 0.5 and self._indices.size > 0:
            return self._attempt_delete
        else:
            return self._attempt_insert

    def _refresh_calc(self):
        """Reattach calculator after atom count changes so it can reinitialize if needed."""
        calc = getattr(self.atoms, "calc", None)
        if calc is not None and hasattr(calc, "set_atoms"):
            calc.set_atoms(self.atoms)

    def _attempt_delete(self, energy: float) -> tuple[bool, float]:
        i = self._rng.choice(self._indices)
        atom = self.atoms.pop(i)
        self._refresh_calc()
        new_energy = float(self.atoms.get_potential_energy())
        delta_e = new_energy - energy
        delta_mu = -self.mus[atom.symbol]
        delta_tot = delta_e + delta_mu

        if self._accept(delta_tot):
            self._counts[atom.symbol] -= 1
            self._count_sum -= 1
            return True, new_energy

        self.atoms.append(atom)
        self._refresh_calc()
        return False, energy

    def _attempt_insert(self, energy: float) -> tuple[bool, float]:
        symbol = self._coerce_symbol(self._rng.choice(self.species))
        position = self._random_position()
        atom = Atom(symbol=symbol, position=position)
        self.atoms.append(atom)
        self._refresh_calc()

        new_energy = float(self.atoms.get_potential_energy())
        delta_e = new_energy - energy
        delta_mu = self.mus[atom.symbol]
        delta_tot = delta_e + delta_mu

        if self._accept(delta_tot):
            self._counts[atom.symbol] += 1
            self._count_sum += 1
            return True, new_energy

        self.atoms.pop(-1)
        self._refresh_calc()
        return False, energy

    def compute(self, step: int):
        if step % self.md_steps != 0:
            return
        
        energy = self._energy
        attempts = self.mc_trials
        accepted = 0
        for _ in range(attempts):
            attempt, energy = self._move()(energy)
            accepted += int(attempt)
            self._update_indices()

        self._update_result(attempts, accepted, energy)