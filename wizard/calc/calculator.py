from itertools import combinations_with_replacement
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import os

from ase import Atoms
from ase.build import cut, rotate, surface
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator
from ase.neb import NEB
from ase.units import J
from calorine.tools import get_elastic_stiffness_tensor

from .phono import PhonoCalc
from ..core.minimize import relax
from ..model.atoms import Morph, AlloyInfo
from ..utils.io import dump_xyz
from ..utils.tools import plot_band_structure

class MaterialCalculator():
    def __init__(self, 
                 atoms: Atoms, 
                 calculator: Calculator, 
                 clamped: bool = False, 
                 **kwargs):
        """
        A wrapper class for performing energy calculations on atomic structures.

        This class writes results to two files for consistency:

        - ``MaterialProperties.xyz``  
        Contains structures generated during calculations (e.g., relaxed
        configurations, defected cells, EOS scaling). Each call appends
        the latest structure in XYZ format.

        - ``MaterialProperties.out``  
        Contains textual property summaries such as lattice constants,
        ground-state energy, defect formation energies, elastic constants,
        and other scalar quantities. Each call appends new results in a
        human-readable format.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object representing the structure.
        calculator : Calculator
            ASE calculator to be attached to the atoms.
        clamped : bool, optional
            If True, skip structural relaxation (default: False).
        **kwargs : dict
            Additional keyword arguments to be handed over to the minimizer; possible arguments can be found
            in the `ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/optimize.html>`_
            https://wiki.fysik.dtu.dk/ase/ase/filters.html#the-frechetcellfilter-class.

        Attributes
        ----------
        atoms : Atoms
            The relaxed (or unrelaxed if clamped=True) structure.
        calc : Calculator
            The calculator used for energy evaluation.
        kwargs : dict
            Relaxation keyword arguments.
        atom_energy : float
            Energy per atom (eV/atom).
        info : str
            Output label composed from formula and config_type.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('Input configuration must be an ASE Atoms object'
                            f', not type {type(atoms)}.')
        if not isinstance(calculator, Calculator):
            raise TypeError('Input calculator must be an ASE Calculator object'
                            f', not type {type(calculator)}.')
        
        atoms = atoms.copy()
        atoms.calc = calculator
        if not clamped:    
            relax(atoms, **kwargs)
        self.atoms = atoms
        self.calc = calculator
        self.clamped = clamped
        self.kwargs = kwargs
        self.atom_energy = atoms.get_potential_energy() / len(atoms)
        formula = atoms.info.get('formula', atoms.get_chemical_formula())
        config_type = atoms.info.get('config_type')
        self.info = f'{formula}_{config_type}' if config_type else formula
        self.symbols = sorted(set(atoms.get_chemical_symbols()))
    
    def isolate_atom_energy(self) -> list[str]:
        """
        Calculate reference energies of isolated atoms for each element type.

        Returns
        -------
        list[str]
            Lines with per-atom reference energies, e.g. "Fe Iso_Atom_Energy: -4.1234 eV".
        """
        output = []
        for symbol in self.symbols:
            atoms = Atoms([symbol], positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
            atoms.calc = self.calc
            atoms.info['formula'] = symbol
            atoms.info['config_type'] = 'isolate_atom'
            iso_atom_energy = atoms.get_potential_energy()
            output.append(f" {symbol:<10}Iso_Atom_Energy: {iso_atom_energy:.4f} eV")
            dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', "a") as f:
            f.write("\n".join(output) + "\n")
        return output
    
    def lattice_constant(self) -> list[str]:
        """
        Report basic lattice properties of the current structure.

        Returns
        -------
        list[str]
            Lines with formatted lattice parameters, angles, ratios, volume,
            and ground-state energy.
        """
        atoms = self.atoms
        atom_energy = self.atom_energy
        a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
        V = atoms.get_volume()               
        N = len(atoms)
        b_over_a = b / a if a > 1e-12 else float('nan')
        c_over_a = c / a if a > 1e-12 else float('nan')
        a_mean = (a + b + c) / 3.0
        output = []
        output.append(f" {self.info:<10}Lattice_Constants: a={a:.4f} Å  b={b:.4f} Å  c={c:.4f} Å")
        output.append(f"{'':<11}Angles: α={alpha:.2f}°  β={beta:.2f}°  γ={gamma:.2f}°")
        output.append(f"{'':<11}Ratios: b/a={b_over_a:.6f}  c/a={c_over_a:.6f}  a_mean={a_mean:.4f} Å")
        output.append(f"{'':<11}Volume: V={V:.3f} Å³  V/atom={V/N:.3f} Å³/atom")
        output.append(f"{'':<11}Ground_State_Energy: {atom_energy:.4f} eV/atom")
        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', "a") as f:
            f.write("\n".join(output) + "\n")
        return output
    
    def elastic_constant(self, epsilon: float = 1e-3) -> dict:
        """
        Compute elastic stiffness constants using the calorine package.

        Parameters
        ----------
        epsilon : float, optional
            Strain amplitude for finite-difference calculation (default 1e-3).

        Returns
        
        dict
        A dictionary with two keys:
        - 'Cij' : np.ndarray
            The full elastic stiffness tensor (in GPa).
        - 'output' : list[str]
            Lines with formatted elastic constants (in GPa), 
        """
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        Cij = get_elastic_stiffness_tensor(atoms, epsilon=epsilon)
        
        output = []
        output.append(f" {self.info:<10}C11: {Cij[0][0]:>7.2f} GPa")
        output.append(f"{'':<11}C12: {Cij[0][1]:>7.2f} GPa")
        output.append(f"{'':<11}C13: {Cij[0][2]:>7.2f} GPa")
        output.append(f"{'':<11}C33: {Cij[2][2]:>7.2f} GPa")
        output.append(f"{'':<11}C44: {Cij[3][3]:>7.2f} GPa")
        output.append(f"{'':<11}C66: {Cij[5][5]:>7.2f} GPa")
        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            f.write("\n".join(output) + "\n")

        return {
            'output': output,
            'Cij': Cij
        }
    
    def eos_curve(self) -> str:
        """
        Generate an equation-of-state (EOS) curve by scaling the unit cell.

        Results:
        - Data written to 'eos_curve_out/{info}_eos_curve.out'
        - Figure saved as 'eos_curve_png/{info}_eos_curve.png'

        Returns
        -------
        str
            Path to the saved EOS curve figure (PNG).
        """
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        os.makedirs('eos_curve_out', exist_ok=True)
        os.makedirs('eos_curve_png', exist_ok=True)
        volumes, energies = [], []
        origin_cell = atoms.cell.copy()
        for scale in np.arange(0.9, 1.20, 0.01):
            atoms.set_cell(scale * origin_cell, scale_atoms = True)
            volumes.append(atoms.get_volume() / len(atoms))
            energies.append(atoms.get_potential_energy() / len(atoms))
            dump_xyz('MaterialProperties.xyz', atoms)
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        font_size = 12
        ax.plot(volumes, energies, '-o')
        ax.set_xlabel('Volume(A$^3$/atom)', fontsize=font_size)
        ax.set_ylabel('Energy (eV/atom)', fontsize=font_size)
        ax.set_title(f'{self.info} EOS Curve', fontsize=font_size)
        fig_path = os.path.join('eos_curve_png',f'{self.info}_eos_curve.png')
        fig.savefig(fig_path)
        plt.close(fig)
        with open(os.path.join('eos_curve_out',f'{self.info}_eos_curve.out'), 'w') as f:
            f.write("Volume(A^3/atom)   Energy(eV/atom)\n")
            for volume, energy in zip(volumes, energies):
                f.write(f"{volume:.2f}   {energy:.4f}\n")
        return fig_path
    
    def dimer_curve(self, distances=np.arange(1.2, 2.8, 0.1)) -> list[str]:
        """
        Generate energy curves for dimers of each element type in the structure.
        For each unique pair of elements (including self-pairs), this method creates a dimer structure, calculates its energy at various interatomic distances, and saves the results.
        Results:
        - Data written to 'dimer_curve_out/{s1}_{s2}_dimer.out' for each pair of elements s1 and s2.
        - Figures saved as 'dimer_curve_png/{s1}_{s2}_dimer.png' for each pair of elements s1 and s2.
        Parameters
        ----------
        distances : array-like, optional
            A sequence of interatomic distances (in Å) at which to evaluate the dimer energy
            (default: np.arange(1.2, 2.8, 0.1)).
        Returns
        -------
        list[str]
            A list of file paths to the saved dimer curve figures (PNG).
        """
        fig_paths = []
        os.makedirs('dimer_curve_out', exist_ok=True)
        os.makedirs('dimer_curve_png', exist_ok=True)

        for s1, s2 in combinations_with_replacement(self.symbols, 2):
            energies = []

            for d in distances:
                atoms = Atoms(symbols=[s1, s2], positions=[[0, 0, 0], [d, 0, 0]],
                              pbc = [True, True, True], cell= [[20,0,0],[0,30,0],[0,0,40]])
                atoms.calc = self.calc
                energy = atoms.get_potential_energy() / len(atoms)
                energies.append(energy)
                dump_xyz("MaterialProperties.xyz", atoms)

            plt.rcParams.update({'font.size': 12}) 
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
            ax.plot(distances, energies, "-o")
            ax.set_xlabel("Distance (Å)")
            ax.set_ylabel("Energy (eV/atom)")
            ax.set_title(f"{s1}-{s2} Dimer")
            fig_path = os.path.join('dimer_curve_png', f'{s1}_{s2}_dimer.png')
            fig.savefig(fig_path)
            plt.close(fig)
            fig_paths.append(fig_path)
            out_path = os.path.join('dimer_curve_out', f'{s1}_{s2}_dimer.out')
            with open(out_path, "w") as f:
                f.write("Distance(A)   Energy(eV)\n")
                for distance, energy in zip(distances, energies):
                    f.write(f"{distance:.2f}   {energy:.4f}\n")
        return fig_paths

    def phonon_dispersion(self, special_points=None, labels_path=None) -> str:
        """
        Calculate and plot the phonon dispersion relation using PhonoCalc.

        Parameters
        ----------
        special_points : dict[str, tuple[float, float, float]] | None
            High-symmetry points in fractional reciprocal coordinates.
            Example for bcc:
                {'G': (0, 0, 0), 
                 'H': (0.5, -0.5, 0.5), 
                 'N': (0, 0, 0.5), 
                 'P': (0.25, 0.25, 0.25)},
        labels_path : list[list[str]] | None
            Band path as a list of label sequences.
            Example for bcc:
                [[['N', 'G', 'H' ,'P', 'G']]]

        Returns
        -------
        str
            Path to the saved phonon dispersion figure (PNG).
        """
        atoms = self.atoms.copy()
        calc = self.calc
        PhonoCalc(atoms, calc).get_band_structure(special_points=special_points, labels_path=labels_path)
        fig_path = plot_band_structure(atoms, self.info)
        return fig_path
             
    def formation_energy_surface(self, hkl = (1, 0, 0), layers = 10, shuffle_symbols = False) -> float:
        """
        Calculate the surface formation energy of a given Miller index (hkl) using the slab method.
        Parameters
        ----------
        hkl : tuple[int, int, int], optional
            Miller index of the surface (default: (1, 0, 0)).
            ASE uses three-index Miller indices. For hcp, common inputs are:
            basal (0001) -> (0, 0, 1), prismatic (10-10) -> (1, 0, 0),
            first-order pyramidal (10-11) -> (1, 0, 1),
            second-order pyramidal (11-22) -> (1, 1, 2).
        layers : int, optional
            Number of atomic layers in the slab (default: 10).
        shuffle_symbols : bool, optional
            Whether to randomly shuffle atomic symbols in the bulk structure before creating the slab (default: False). 
            This can be useful for disordered materials to get a more representative surface energy.
        Returns
        -------
        float
            Surface formation energy in J/m^2.
        """
        atoms = self.atoms.copy()
        bulk = surface(atoms, hkl, layers = layers, vacuum=0)
        bulk.pbc = [True, True, True]
        if shuffle_symbols:
            Morph(bulk).shuffle_symbols()
        bulk.calc = self.calc
        relax(bulk, constant_cell=True, **self.kwargs)

        slab = bulk.copy()
        slab.pbc = [True, True, False]
        slab.center(vacuum=10, axis=2)
        slab.calc = self.calc
        relax(slab, constant_cell=True, **self.kwargs)
        box = slab.get_cell()
        S = np.linalg.norm(np.cross(box[0], box[1]))
        bulk_energy = bulk.get_potential_energy()
        slab_energy = slab.get_potential_energy()
        formation_energy = (slab_energy - bulk_energy) / S / 2

        hkl_str = '-'.join(map(str, hkl))
        dump_xyz('MaterialProperties.xyz', bulk)
        dump_xyz('MaterialProperties.xyz', slab)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.info:<10}{hkl_str} Surface_Energy: {formation_energy / J / 1e-20 :.4f} J/m^2', file=f)
        return formation_energy * 1000
    
    def _formation_energy_defect(self, atoms : Atoms) -> float:
        atoms.calc = self.calc
        relax(atoms, **self.kwargs)
        return atoms.get_potential_energy() - self.atom_energy * len(atoms)

    def formation_energy_vacancy(self, index = 0) -> float:
        """
        Calculate the vacancy formation energy by removing an atom at the specified index and relaxing the structure.
        Parameters
        ----------
        index : int, optional
            Index of the atom to remove for vacancy formation (default: 0).
        Returns
        -------
        float
            mono-vacancy formation energy in eV.
        """
        atoms = self.atoms.copy()
        Morph(atoms).create_vacancy(index = index)
        formation_energy = self._formation_energy_defect(atoms)
        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.info:<10}Formation_Energy_Vacancy: {formation_energy:.4f} eV', file=f)
        return formation_energy

    def formation_energy_divacancies(self, index0 = 0, index1 = 1) -> float:
        """
        Calculate the divacancy formation energy by removing two atoms at the specified indices and relaxing the structure.
        Parameters
        ----------
        index1 : int, optional
            Index of the first atom to remove for divacancy formation (default: 0).
        index2 : int, optional
            Index of the second atom to remove for divacancy formation (default: 1).
        Returns
        -------
        float
            mono-vacancy formation energy in eV.
        """
        atoms = self.atoms.copy()
        index0, index1 = sorted((index0, index1))
        Morph(atoms).create_vacancy(index = index1)
        Morph(atoms).create_vacancy(index = index0)
        formation_energy = self._formation_energy_defect(atoms)
        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.info:<10}Formation_Energy_Divacancies: {formation_energy:.4f} eV', file=f)
        return formation_energy

    def migration_energy_vacancy(self, index0 = 0, index1 = 1) -> list[float]:
        """
        Calculate the migration energy of a vacancy by performing a NEB calculation.
        Parameters
        ----------
        index0 : int, optional
            Index of the atom to be removed for the initial vacancy (default: 0).
        index1 : int, optional
            Index of the neighboring site to which the vacancy migrates (default: 1).
        Returns
        -------
        float      
            Vacancy migration energy in eV.
        """

        os.makedirs('migration_energy_vacancy', exist_ok=True)
        atoms = self.atoms.copy() 
        symbol = atoms[index0].symbol
        atoms[index1].symbol = symbol
        initial = atoms.copy()
        del initial[index0]
        final = atoms.copy()
        del final[index1]

        initial.calc = self.calc
        relax(initial, **self.kwargs)
        cell = initial.get_cell()
        final.set_cell(cell)
        final.calc = self.calc
        relax(final, **self.kwargs, constant_cell=True)

        images = [initial] + [initial.copy() for _ in range(11)] + [final]
        for i in images:
            i.calc = self.calc
        neb = NEB(images, climb=True, allow_shared_calculator=True)
        neb.interpolate()
        try:
            relax(neb, **self.kwargs, constant_cell=True)
        except Exception as exc:
            raise RuntimeError("Failed to relax vacancy migration NEB images.") from exc
        energies = [image.get_potential_energy() for image in images]
        energies = np.array(energies)
        migration_energy = max(energies) - min(energies)
        energies -= min(energies)
        for image in images:
            dump_xyz('MaterialProperties.xyz', image)  

        plt.rcParams.update({'font.size': 12}) 
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        ax.plot(np.linspace(0, 1, len(energies)), energies, '-o')
        ax.set_xlabel('Reaction Coordinate')
        ax.set_ylabel('Energy (eV)')
        ax.set_title(f'{self.info} {symbol}-Vacancy Migration Energy')
        fig_path = os.path.join('migration_energy_vacancy', f'{self.info}_{symbol}-vacancy_migration_energy.png')
        fig.savefig(fig_path)   
        plt.close(fig)

        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.info:<10}Migration_Energy_({symbol}-Vacancy): {migration_energy:.4f} eV', file=f)
        
        return energies
    
    def formation_energy_sia(self, vector = (1, 1, 1), index = 0) -> float:
        """
        Calculate the self-interstitial atom (SIA) formation energy by adding an atom at the specified index.
        Parameters
        ----------
        vector : tuple[int, int, int], optional
            Relative position vector (in lattice units) from the original atom to the interstitial site (default: (1, 0, 0)).
        index : int, optional
            Index of the original atom to which the interstitial atom is added (default: 0).
        Returns
        -------
        float
            Self-interstitial atom formation energy in eV.
        """
        atoms = self.atoms.copy()
        Morph(atoms).create_self_interstitial_atom(vector, index = index)
        formation_energy = self._formation_energy_defect(atoms)
        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.info:<10}{vector} Formation_Energy_Sia: {formation_energy:.4} eV', file=f)
        return formation_energy
    
class AlloyCalculator(MaterialCalculator):
    def __init__(self, 
                 alloy_info: AlloyInfo,
                 calculator: Calculator,
                 supercell: tuple = (3, 3, 3), 
                 clamped: bool = False,
                 **kwargs): 
        if not isinstance(alloy_info, AlloyInfo):
            raise TypeError('Input alloy_info must be an AlloyInfo object'
                            f', not type {type(alloy_info)}.')
        
        self.alloy_info = alloy_info
        atoms = alloy_info.create_bulk_atoms(supercell = supercell)
        super().__init__(atoms, calculator, clamped, **kwargs)

    def stacking_fault(self, a, b, miller, distance):
        '''
        ---------------------------------------------------------------------------------------------------
        For FCC-Al                  |   For BCC-Nb
        surf. I        surf. II     |   surf. I              surf. II            surf. III
        (-1, 1,  0)    (-1, 1,  0)  |   (-1, 1, -2)          ( 1,-1,  0)         ( 1, -4/5, 1/5)
        ( 1, 1, -2)    ( 1, 1,  0)  |   (-1, 1,  1) <111>    ( 1, 1, -1) <111>   ( 1,  1,    -1)  <111>
        ( 1, 1,  1)    ( 0, 0,  1)  |   ( 1, 1,  0) {110}    ( 1, 1,  2) {112}   ( 1,  2,     3)  {123}
                                    
        ---------------------------------------------------------------------------------------------------
        For HCP-Ti 
        for basal {0001} Normal--[0001]          for prism {10-10} Normal--[10-10]              
        uvws =  [[-2,1,1,0],  [[-1, 0,0]          uvws =  [[-1,2,-1, 0],    [[0,1,0] 
                 [0,-1,1,0],   [-1,-2,0]                   [0, 0, 0, 1],     [0,0,1] 
                 [0, 0,0,1]]   [ 0, 0,1]]                  [1, 0,-1, 0]]     [2,1,0]] 
             
        ---------------------------------------------------------------------------------------------------                                                       
        for Pyramidal I narrow  {10-11} Normal: None    for Pyramidal I wide {10-11} Normal: None   
        uvws = [[-1, 2,-1, 0],  [0,  1,0]               uvws = [[-1,-1,  2, 3],     [[-1,-1,1]      
                [-1, 0, 1, 2],  [-2,-1,2]                       [-1, 0,  1, 2],      [-2,-1,2]      
                 {1,  0,-1, 1}]                                  {1,  0, -1, 1}]                 
                                                                                                                                                         
        for Pyramidal II {11-22} Normal       
        uvws =  [[-1,-1, 2,3],     [[-1,-1,1]
                [-1, 1, 0,0],      [-1, 1,0]
                {1,1,-2,  2}       [N,  N,N]]                                                                         
        ---------------------------------------------------------------------------------------------------
        Hexagonal Miller direction indices to Miller-Bravais indices and back:
        [-1,-1,2,3] = [-1,-1,1]
        [-1, 1,0,0] = [-1, 1,0]
        ---------------------------------------------------------------------------------------------------
        ''' 
        atoms = self.atoms.copy()
        atoms.calc = self.calc

        slab = cut(atoms, a, b, clength=None, origo=(0,0,0), nlayers = 18, extend=1.0, tolerance=0.01, maxatoms=None)
        rotate(slab, a, (1,0,0), b, (0,1,0), center=(0,0,0))
        slab.calc = self.calc
        relax(slab, constant_cell=True, **self.kwargs)
        slab.center(axis=2)
        slab.constraints = FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < 1/2 * slab.cell[2][2]])
        box = slab.get_cell()
        S = np.linalg.norm(np.cross(box[0], box[1])) * 2

        shift_distance = np.linalg.norm(np.array(a)) * distance
        shift_indices = [atom.index for atom in slab if atom.position[2] > 1/2 * slab.cell[2][2]]
        slide_steps = shift_distance / 10
        
        energies = []
        for i in range(11):
            slab_shift = slab.copy()
            slab_shift.positions[shift_indices] += [slide_steps * i, 0, 0]
            slab_shift.calc = self.calc
            relax(slab_shift, constant_cell=True, **self.kwargs)
            defects_energy = slab_shift.get_potential_energy() / S
            energies.append(defects_energy)
            dump_xyz('MaterialProperties.xyz', slab_shift)

        energies = np.array(energies)
        energies -= energies[0]
        miller_str = '-'.join(map(str, miller))
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.info:<7}{miller_str} Stacking_Fault: {max(energies) * 1000:.4f} meV/A^2', file=f)

        plt.plot(np.linspace(0, 1, len(energies)), energies, marker='o', label=f'{self.info}')  
        plt.legend()
        plt.savefig(f'{self.info}_{miller_str}_stacking_fault.png')
        plt.close()
        return energies
    
    def _bcc_metal_screw_move(self, final_model, energy_divisor = 1.0, image_count = 15):
        initial = self.alloy_info.create_screw_atoms(model = 'initial')
        final = self.alloy_info.create_screw_atoms(model = final_model)
        final.set_chemical_symbols(initial.get_chemical_symbols())

        initial.calc = self.calc
        relax(initial, **self.kwargs)
        final.set_cell(initial.get_cell(), scale_atoms=True)
        final.calc = self.calc
        relax(final, constant_cell=True, **self.kwargs)

        images = [initial] + [initial.copy() for _ in range(image_count)] + [final]
        for image in images:
            image.calc = self.calc
        neb = NEB(images, climb=True, allow_shared_calculator=True)
        neb.interpolate()
        try:
            relax(neb, constant_cell=True, **self.kwargs)
        except Exception as exc:
            raise RuntimeError(f'Failed to relax {final_model} screw NEB images.') from exc

        energies = np.array([image.get_potential_energy() for image in images]) / energy_divisor
        migration_energy = max(energies) - min(energies)
        energies -= min(energies)
        for image in images:
            dump_xyz('MaterialProperties.xyz', image)  

        formula = final.info.get('formula', final.get_chemical_formula())
        config_type = final.info.get('config_type', f'bcc_{final_model}_screw')
        info = f'{formula}_{config_type}' if config_type else formula
        coords = np.linspace(0, 1, len(energies))
        with open('MaterialProperties.out', 'a') as f:
            print(f' {info:<10}Screw_Migration_Energy: {migration_energy:.4f} eV', file=f)

        plt.plot(coords, energies, marker='o', label=f'{info}')
        plt.legend()
        plt.savefig(f'{info}_screw_migration.png')
        plt.close()
        return energies
    
    def bcc_metal_screw_dipole_move(self, image_count = 15):
        return self._bcc_metal_screw_move('dipole_move', energy_divisor=2.0, image_count=image_count)
    
    def bcc_metal_screw_one_move(self, image_count = 15):
        return self._bcc_metal_screw_move('one_move', image_count=image_count)

    
