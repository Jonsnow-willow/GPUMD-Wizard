from ase import Atoms
from ase.mep import NEB
from ase.build import surface
from ase.units import J
from .io import dump_xyz
from .tools import plot_band_structure
from .phono import PhonoCalc
from .atoms import Morph
from calorine.tools import get_elastic_stiffness_tensor, relax_structure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
import os

class MaterialCalculator():
    def __init__(self, atoms, calculator, symbol_info, clamped = False, **kwargs):
        atoms.calc = calculator
        if not clamped:
            relax_structure(atoms, kwargs)
        self.atoms = atoms
        self.calc = calculator  
        self.epa = atoms.get_potential_energy() / len(atoms)
        self.formula = symbol_info.formula
        self.symbols = symbol_info.symbols
        self.crystalstructure = symbol_info.lattice_type
        self.lc = symbol_info.lattice_constant
        self.kwargs = kwargs
    
    def isolate_atom_energy(self):
        output = f"\n {self.formula:<10}Isolated_Atom_Energies (eV):\n"
        for symbol in self.symbols:
            atoms = Atoms(symbols=[symbol], positions=[[0, 0, 0]])
            atoms.calc = self.calc
            iso_atom_energy = atoms.get_potential_energy()
            output += f"   {symbol:<2} Atom Energy: {iso_atom_energy:.4f} eV\n"
        with open('MaterialProperties.out', "a") as f:
            f.write(output)
        return output

    def lattice_constant(self):
        atoms = self.atoms
        energy_per_atom = self.epa
        cell_lengths = atoms.cell.cellpar()
        dump_xyz('MaterialProperties.xyz', atoms)
        
        output = ""
        if self.crystalstructure == 'hcp':
            output += f" {self.formula:<10}Lattice_Constants: a: {cell_lengths[0]:.4f} A    c: {cell_lengths[2]:.4f} A\n"
            output += f"{'':<11}Ground_State_Energy: {energy_per_atom:.4f} eV\n"
        else:
            output += f" {self.formula:<10}Lattice_Constants: {round(sum(cell_lengths[:3])/3, 3):.4f} A\n"
            output += f"{'':<11}Ground_State_Energy: {energy_per_atom:.4f} eV\n"
        
        with open('MaterialProperties.out', "a") as f:
            f.write(output)
        
        return output
    
    def elastic_constant(self, epsilon = 0.01):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        Cij = get_elastic_stiffness_tensor(atoms, epsilon=epsilon, **self.kwargs)
        dump_xyz('MaterialProperties.xyz', atoms)
        
        output = ""
        output += f" {self.formula:<10}C11: {Cij[0][0]:>7.2f} GPa\n"
        output += f"{'':<11}C12: {Cij[0][1]:>7.2f} GPa\n"
        output += f"{'':<11}C13: {Cij[0][2]:>7.2f} GPa\n"
        output += f"{'':<11}C33: {Cij[2][2]:>7.2f} GPa\n"
        output += f"{'':<11}C44: {Cij[3][3]:>7.2f} GPa\n"
        output += f"{'':<11}C66: {Cij[5][5]:>7.2f} GPa\n"
        
        with open('MaterialProperties.out', 'a') as f:
            f.write(output)
        
        return output
    
    def eos_curve(self):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        os.makedirs('eos_curve_out', exist_ok=True)
        os.makedirs('eos_curve_png', exist_ok=True)
        volumes, energies = [], []
        origin_cell = atoms.cell.copy()
        for scale in np.arange(0.9, 1.10, 0.01):
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
        ax.set_title(f'{self.formula} {self.crystalstructure} EOS Curve', fontsize=font_size)
        fig_path = os.path.join('eos_curve_png',f'{self.formula}_eos_curve.png')
        fig.savefig(fig_path)
        plt.close(fig)
        with open(os.path.join('eos_curve_out',f'{self.formula}_eos_curve.out'), 'w') as f:
            f.write("Volume(A^3/atom)   Energy(eV/atom)\n")
            for volume, energy in zip(volumes, energies):
                f.write(f"{volume:.2f}   {energy:.4f}\n")

        return fig_path

    def phonon_dispersion(self, special_points = None, labels_path = None):
        atoms = self.atoms.copy()
        calc = self.calc
        PhonoCalc(atoms, calc).get_band_structure(special_points=special_points, labels_path=labels_path)
        fig_path = plot_band_structure(atoms, self.formula, self.crystalstructure)
        return fig_path
             
    def formation_energy_surface(self, hkl = (1, 0, 0), layers = 10):
        atoms = self.atoms.copy()
        energy_per_atom = self.epa
        slab = surface(atoms, hkl, layers = layers, vacuum=10) 
        Morph(slab).shuffle_symbols()
        slab.calc = self.calc
        relax_structure(slab, **self.kwargs)
        box = slab.get_cell()
        S = (box[0][0] * box[1][1] - box[0][1] * box[1][0])
        slab_energy = slab.get_potential_energy()
        formation_energy = (slab_energy - energy_per_atom * len(slab)) / S / 2

        if self.crystalstructure == 'hcp':
            hk, l = hkl[:2], hkl[2]
            hkl_str = '-'.join(map(str, sorted(hk, reverse=True)))
            hkl_str += f'-{l}'
        else:
            hkl_str = '-'.join(map(str, sorted(hkl, reverse=True)))

        dump_xyz('MaterialProperties.xyz', slab)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}{hkl_str} Surface_Energy: {formation_energy / J / 1e-20 :.4f} J/m^2', file=f)
        return formation_energy * 1000

    def formation_energy_vacancy(self, index = 0):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        energy_per_atom = self.epa
        Morph(atoms).create_vacancy(index = index)
        relax_structure(atoms, **self.kwargs)
        formation_energy = atoms.get_potential_energy() - energy_per_atom * len(atoms)

        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}Formation_Energy_Vacancy: {formation_energy:.4f} eV', file=f)
        return formation_energy

    def migration_energy_vacancy(self, index0 = 0, index1 = 1):
        atoms = self.atoms.copy() 
        symbol = atoms[index0].symbol
        atoms[index1].symbol = symbol
        initial = atoms.copy()
        del initial[index0]
        final = atoms.copy()
        del final[index1]

        initial.calc = self.calc
        relax_structure(initial, **self.kwargs)
        relax_cell = initial.get_cell()
        final.set_cell(relax_cell)
        final.calc = self.calc
        relax_structure(final, **self.kwargs, constant_cell=True)

        images = [initial] + [initial.copy() for _ in range(11)] + [final]
        for i in images:
            i.calc = self.calc
        neb = NEB(images, climb=True, allow_shared_calculator=True)
        neb.interpolate()
        relax_structure(neb, **self.kwargs)
        energies = [image.get_potential_energy() for image in images]
        energies = np.array(energies)
        migration_energy = max(energies) - min(energies)
        energies -= min(energies)
        for image in images:
            dump_xyz('MaterialProperties.xyz', image)  
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}Migration_Energy_({symbol}-Vacancy): {migration_energy:.4f} eV', file=f)
        plt.plot(np.linspace(0, 1, len(energies)), energies, marker='o', label=f'{self.formula}')  
        plt.legend()
        plt.savefig(f'{self.formula}_migration_{symbol}_vacancy.png')
        plt.close()
        return energies
    
    def formation_energy_sia(self, vector = (1, 0, 0), index = 0):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        energy_per_atom = self.epa
        Morph(atoms).create_self_interstitial_atom(vector, index = index)
        relax_structure(atoms, **self.kwargs)
        formation_energy = atoms.get_potential_energy() - energy_per_atom * len(atoms)

        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}{vector} Formation_Energy_Sia: {formation_energy:.4} eV', file=f)
        return formation_energy