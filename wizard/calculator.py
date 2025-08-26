from ase import Atoms, Atom
from ase.build import cut, rotate
from ase.optimize import FIRE
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.build import surface
from ase.units import J
from wizard.io import get_nth_nearest_neighbor_index, relax, dump_xyz, read_xyz, plot_band_structure
from calorine.tools import get_elastic_stiffness_tensor
from wizard.phono import PhonoCalc
from wizard.atoms import Morph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from importlib.resources import files
import tempfile
import numpy as np
import os
import re

class MaterialCalculator():
    def __init__(self, atoms, calculator, symbol_info):
        atoms.calc = calculator
        relax(atoms)
        atom_energy = atoms.get_potential_energy() / len(atoms)
        self.atom_energy = atom_energy
        self.atoms = atoms
        self.calc = calculator  
        self.formula = symbol_info.formula
        self.crystalstructure = symbol_info.lattice_type
        self.lc = symbol_info.lattice_constant
    
    def isolate_atom_energy(self):
        symbol = self.atoms.get_chemical_symbols()[0]
        atoms = Atoms(symbols = [symbol], positions=[[0,0,0]])
        atoms.calc = self.calc
        iso_atom_energy = atoms.get_potential_energy()
        with open('MaterialProperties.out', "a") as f:
            f.write(f" {self.formula:<7}Atom_Energy: {iso_atom_energy:.4f} eV\n")
        return iso_atom_energy

    def get_potential_energy(self):
        energy = self.atom_energy
        return energy

    def lattice_constant(self):
        atoms = self.atoms
        atom_energy = self.atom_energy
        cell_lengths = atoms.cell.cellpar()
        dump_xyz('MaterialProperties.xyz', atoms)
        
        output = ""
        if self.crystalstructure == 'hcp':
            output += f" {self.formula:<10}Lattice_Constants: a: {cell_lengths[0]:.4f} A    c: {cell_lengths[2]:.4f} A\n"
            output += f"{'':<11}Ground_State_Energy: {atom_energy:.4f} eV\n"
        else:
            output += f" {self.formula:<10}Lattice_Constants: {round(sum(cell_lengths[:3])/3, 3):.4f} A\n"
            output += f"{'':<11}Ground_State_Energy: {atom_energy:.4f} eV\n"
        
        with open('MaterialProperties.out', "a") as f:
            f.write(output)
        
        return output
    
    def elastic_constant(self, epsilon = 0.01):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        Cij = get_elastic_stiffness_tensor(atoms, epsilon=epsilon)
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
             
    def formation_energy_surface(self, hkl = (1, 0, 0), layers = 10, relax_params = {}):
        atoms = self.atoms.copy()
        atom_energy = self.atom_energy
        slab = surface(atoms, hkl, layers = layers, vacuum=10) 
        Morph(slab).shuffle_symbols()
        slab.calc = self.calc
        relax(slab, **relax_params)
        box = slab.get_cell()
        S = (box[0][0] * box[1][1] - box[0][1] * box[1][0])
        slab_energy = slab.get_potential_energy()
        formation_energy = (slab_energy - atom_energy * len(slab)) / S / 2

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

    def formation_energy_vacancy(self, index = 0, relax_params = {}):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        atom_energy = self.atom_energy
        Morph(atoms).create_vacancy(index = index)
        relax(atoms, **relax_params)
        formation_energy = atoms.get_potential_energy() - atom_energy * len(atoms)

        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}Formation_Energy_Vacancy: {formation_energy:.4f} eV', file=f)
        return formation_energy

    def formation_energy_divacancies(self, nth = 1, index = 0, relax_params = {}):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        atom_energy = self.atom_energy
        Morph(atoms).create_divacancies(index1= index, nth = nth)
        relax(atoms, **relax_params)
        formation_energy = atoms.get_potential_energy() - atom_energy * len(atoms)

        dump_xyz('MaterialProperties.xyz', atoms)     
        with open('MaterialProperties.out', 'a') as f:
            f.write(f' {self.formula:<7}{nth}th Formation_Energy_Divacancies: {formation_energy:.4f} eV\n')
        return formation_energy

    def migration_energy_vacancy(self, index0 = 0, index1 = 1, fmax = 0.02, steps = 500):
        atoms = self.atoms.copy() 
        symbol = atoms[index0].symbol
        atoms[index1].symbol = symbol
        initial = atoms.copy()
        del initial[index0]
        final = atoms.copy()
        del final[index1]

        initial.calc = self.calc
        relax(initial)
        relax_cell = initial.get_cell()
        final.set_cell(relax_cell)
        final.calc = self.calc
        relax(final, method='ucf')

        images = [initial] + [initial.copy() for _ in range(11)] + [final]
        for i in images:
            i.calc = self.calc
        neb = NEB(images,climb=True, allow_shared_calculator=True)
        neb.interpolate()
        
        optimizer = FIRE(neb)
        optimizer.run(fmax=fmax, steps=steps)
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
    
    def formation_energy_sia(self, vector = (1, 0, 0), index = 0, relax_params = {}):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        atom_energy = self.atom_energy
        Morph(atoms).create_self_interstitial_atom(vector, index = index)
        relax(atoms, **relax_params)
        formation_energy = atoms.get_potential_energy() - atom_energy * len(atoms)

        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}{vector} Formation_Energy_Sia: {formation_energy:.4} eV', file=f)
        return formation_energy
    
    def migration_energy_sia(self, vector1, vector2, fmax=0.02, steps=500):
        atoms = self.atoms.copy() 
        initial = atoms.copy()
        final = atoms.copy()
        index = get_nth_nearest_neighbor_index(initial, 0, 1)
        Morph(initial).create_self_interstitial_atom(vector1, index = 0)
        Morph(final).create_self_interstitial_atom(vector2, index = index)

        initial.calc = self.calc
        relax(initial)
        relax_cell = initial.get_cell()
        final.set_cell(relax_cell)
        final.calc = self.calc
        relax(final, method='ucf')

        images = [initial] + [initial.copy() for i in range(11)] + [final]
        for i in images:
            i.calc = self.calc
        neb = NEB(images, allow_shared_calculator=True)
        neb.interpolate()
        
        optimizer = FIRE(neb)
        optimizer.run(fmax=fmax, steps=steps)
        energies = [image.get_potential_energy() for image in images]
        energies = np.array(energies)
        migration_energy = max(energies) - min(energies)
        energies -= min(energies)
        for image in images:
            dump_xyz('MaterialProperties.xyz', image)  
        with open('MaterialProperties.out', 'a') as f:
            print(f'{self.formula:^4}   Migration_Energy_SIA: {migration_energy:.4f} eV', file=f)
        plt.plot(np.linspace(0, 1, len(energies)), energies, marker='o')  
        plt.savefig(f'migration_SIA.png')
        plt.close()
        return energies
    
    def formation_energy_interstitial_atom(self, symbol, fractional_position, config_type, new_atom_e = 0, relax_params = {}):
        atoms = self.atoms.copy() 
        atoms.calc = self.calc
        atoms_energy = atoms.get_potential_energy()
        if new_atom_e == 0:
            new_atom_e = self.atom_energy
        position = np.dot(fractional_position, self.atoms.get_cell())
        insert_atom = Atom(symbol, position)
        atoms.append(insert_atom)
        atoms.calc = self.calc
        relax(atoms, **relax_params)
        formation_energy = atoms.get_potential_energy() - atoms_energy - new_atom_e

        dump_xyz('MaterialProperties.xyz', atoms)
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}{config_type} Formation_Energy: {formation_energy:.4f} eV', file=f)
        return formation_energy
    
    def migration_energy_interstitial(self, symbols, fractional_position, config_type, fmax = 0.02, steps = 500):
        atoms = self.atoms.copy()
        position0 = np.dot(fractional_position[0], self.atoms.get_cell())
        position1 = np.dot(fractional_position[1], self.atoms.get_cell())
        insert_atom1 = Atom(symbols[0], position0)
        insert_atom2 = Atom(symbols[1], position1)
        initial = atoms.copy()
        initial.append(insert_atom1)
        final = atoms.copy()
        final.append(insert_atom2)

        initial.calc = self.calc
        relax(initial)
        relax_cell = initial.get_cell()
        final.set_cell(relax_cell)
        final.calc = self.calc
        relax(final, method='ucf')

        images = [initial] + [initial.copy() for i in range(11)] + [final]
        for i in images:
            i.calc = self.calc
        neb = NEB(images, allow_shared_calculator=True)
        neb.interpolate()
        
        optimizer = FIRE(neb)
        optimizer.run(fmax=fmax, steps=steps)
        energies = [image.get_potential_energy() for image in images]
        energies = np.array(energies)
        migration_energy = max(energies) - min(energies)
        energies -= min(energies)
        for image in images:
            dump_xyz('MaterialProperties.xyz', image)  
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}   Migration_Energy_{config_type}: {migration_energy:.4f} eV', file=f)
        plt.plot(np.linspace(0, 1, len(energies)), energies, marker='o', label=f'{config_type}')  
        plt.legend()
        plt.savefig(f'migration_{config_type}.png')
        plt.close()
        return energies
    
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
        rotate(slab, a,(1,0,0),b,(0,1,0), center=(0,0,0))
        slab.calc = self.calc
        relax(slab)
        slab.center(axis=2)
        slab.constraints = FixAtoms(indices=[atom.index for atom in slab if atom.position[2] < 1/2 * slab.cell[2][2]])
        box = slab.get_cell()
        S = (box[0][0] * box[1][1] - box[0][1] * box[1][0]) * 2

        shift_distance = np.linalg.norm(np.array(a)) * distance
        shift_indices = [atom.index for atom in slab if atom.position[2] > 1/2 * slab.cell[2][2]]
        slide_steps = shift_distance / 10
        
        energies = []
        for i in range(11):
            slab_shift = slab.copy()
            slab_shift.positions[shift_indices] += [slide_steps * i, 0,0]
            slab_shift.calc = self.calc
            relax(slab_shift, method='fixed_line')
            defects_energy = slab_shift.get_potential_energy() / S
            energies.append(defects_energy)
            dump_xyz('MaterialProperties.xyz', slab_shift)

        energies = [e - energies[0] for e in energies]
        with open('MaterialProperties.out', 'a') as f:
            print(f' {self.formula:<7}{miller} Stacking_Fault: {max(energies) * 1000:.4f} meV/A^2', file=f)

        plt.plot(np.linspace(0, 1, len(energies)), energies, marker='o', label=f'{self.formula}')  
        plt.legend()
        plt.savefig(f'{self.formula}_stacking_fault_{miller}.png')
        plt.close()
        return energies
    
    def bcc_metal_screw_dipole_move(self, fmax = 0.02, steps = 500):
        lc = self.lc
        symbols = []
        compositions = []
        for symbol, composition in re.findall('([A-Z][a-z]*)(\d*)', self.formula):
            symbols.append(symbol)
            compositions.append(int(composition) if composition else 1)
        if len(symbols) > 1:
            element_ratio = np.array(compositions) / sum(compositions)
            element_counts = np.ceil(element_ratio * 135).astype(int)
            symbols = np.repeat(symbols, element_counts)
            np.random.shuffle(symbols)
            sym = symbols[:135]
        else:
            sym = [symbols[0] for _ in range(135)]

        initial_screw = read_xyz(str(files('wizard.str').joinpath('Fe_screw.xyz')))
        for i in initial_screw:
            i.set_chemical_symbols(sym)
            i.calc = self.calc

        initial = initial_screw[0]
        final = initial_screw[1]

        unit_cell = initial.cell.copy()
        initial.set_cell(lc * unit_cell, scale_atoms=True)
        relax(initial, method='ucf')
        initial_cell = initial.cell.copy()
        final.set_cell(initial_cell, scale_atoms=True)
        relax(final, method='ucf')

        images = [initial] + [initial.copy() for i in range(15)] + [final]
        for i in images:
            i.calc = self.calc
        neb = NEB(images, allow_shared_calculator=True)
        neb.interpolate()    
        optimizer = FIRE(neb)
        optimizer.run(fmax=fmax, steps=steps)
        energies = [image.get_potential_energy() / 2  for image in images]
        energies = np.array(energies)
        energies -= min(energies)
        for image in images:
            dump_xyz('MaterialProperties.xyz', image)  

        plt.plot(np.linspace(0, 1, len(energies)), energies, marker='o', label=f'{self.formula}')
        plt.legend()
        plt.savefig(f'{self.formula}_screw_dipole_move.png')
        plt.close()
        return energies

    def bcc_metal_screw_one_move(self, fmax = 0.02, steps = 500):
        lc = self.lc
        symbols = []
        compositions = []
        for symbol, composition in re.findall('([A-Z][a-z]*)(\d*)', self.formula):
            symbols.append(symbol)
            compositions.append(int(composition) if composition else 1)
        if len(symbols) > 1:
            element_ratio = np.array(compositions) / sum(compositions)
            element_counts = np.ceil(element_ratio * 135).astype(int)
            symbols = np.repeat(symbols, element_counts)
            np.random.shuffle(symbols)
            sym = symbols[:135]
        else:
            sym = [symbols[0] for _ in range(135)]
        
        initial_screw = read_xyz(str(files('wizard.str').joinpath('Fe_screw.xyz')))
        for i in initial_screw:
            i.set_chemical_symbols(sym)
            i.calc = self.calc

        initial = initial_screw[0]
        final = initial_screw[2]

        unit_cell = initial.cell.copy()
        initial.set_cell(lc * unit_cell, scale_atoms=True)
        relax(initial, method='ucf')
        initial_cell = initial.cell.copy()
        final.set_cell(initial_cell, scale_atoms=True)
        relax(final, method='ucf')

        images = [initial] + [initial.copy() for i in range(15)] + [final]
        for i in images:
            i.calc = self.calc
        neb = NEB(images, allow_shared_calculator=True)
        neb.interpolate()    
        optimizer = FIRE(neb)
        optimizer.run(fmax=fmax, steps=steps)
        energies = [image.get_potential_energy() for image in images]
        energies = np.array(energies)
        energies -= min(energies)
        for image in images:
            dump_xyz('MaterialProperties.xyz', image)  

        plt.plot(np.linspace(0, 1, len(energies)), energies, marker='o', label=f'{self.formula}')
        plt.legend()
        plt.savefig(f'{self.formula}_screw_one_move.png')
        plt.close()
        return energies