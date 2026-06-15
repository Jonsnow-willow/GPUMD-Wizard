from wizard.utils.io import dump_xyz
from wizard.structure.atoms import Morph
from calorine.tools import relax_structure
import numpy as np 
import random 
import re

class MultiMol(): 
    
    def __init__(self, frames):         
        self.frames = frames

    def select_by_size(self, num):
        select_set = []
        for atoms in self.frames:
            if len(atoms) == num:
                select_set.append(atoms)
        return select_set
    
    def select_by_symbols(self, *symbols):
        select_set = []
        for atoms in self.frames:
            chemical_symbols = atoms.get_chemical_symbols()
            if any(chemical_symbol not in symbols for chemical_symbol in chemical_symbols):
                continue
            else:
                select_set.append(atoms)
        return select_set
    
    def select_by_num_of_symbols(self, num):
        select_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            unique_elements = set(s)
            if len(unique_elements) == num:
                select_set.append(atoms)
        return select_set

    def select_by_formulars(self, *formulars):
        select_set = []
        symbols_lists = []
        for formula in formulars:
            symbols_lists.append(re.findall('([A-Z][a-z]*)', formula))
        for atoms in self.frames:
            symbols = atoms.get_chemical_symbols()
            for symbols_list in symbols_lists:
                if set(symbols) == set(symbols_list):
                    select_set.append(atoms)
                    break
        return select_set
    
    def select_by_config_type(self, config_type):
        select_set = []
        for atoms in self.frames:
            if atoms.info['config_type'] == config_type:
                select_set.append(atoms)
        return select_set
    
    def select_by_1th_distance(self, min = 3.5, max = 100):
        select_set = []
        for atoms in self.frames:
            distances = atoms.get_all_distances(mic=True)
            non_zero_distances = distances[distances > 0]
            if non_zero_distances.size == 0:
                continue
            min_distance = np.min(non_zero_distances)
            if min_distance > min and min_distance < max:
                select_set.append(atoms)
        return select_set
    
    def select_by_force(self, force_min, force_max):
        select_set = []
        for atoms in self.frames:
            f = np.concatenate(atoms.info['forces'])
            if np.all((f > force_min) & (f < force_max)):
                select_set.append(atoms)
        return select_set
    
    def select_random(self, num):
        select_set = []
        select_set = random.sample(self.frames, num)
        return select_set
    
    def select_by_force_error(self, calc, error_min = 0, error_max = 100):
        select_set = []
        for atoms in self.frames:
            atoms.calc = calc
            f_1 = np.concatenate(atoms.get_forces())
            f_2 = np.concatenate(atoms.info['forces'])
            diff = abs(f_1 - f_2)
            if np.any((diff > error_min) & (diff < error_max)):
                select_set.append(atoms)
        return select_set
    
    def select_by_coherent_energy(self, calc, coh = {}, e_min = 0, emax = 10000):
        select_set = []
        for atoms in self.frames:
            if len(set(atoms.symbols)) == 1:
                continue
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            for atom in atoms:
                energy -= coh[atom.symbol]
            if energy > e_min and energy < emax:
                select_set.append(atoms)
        return select_set
       
    def active_learning(self, calcs, error_min = 0, error_max = 100, n = 1000000):
        Train_set = []
        for atoms in self.frames:
            forces = []
            for calc in calcs:
                atoms.calc = calc
                forces.append(atoms.get_forces())
            forces = np.std(forces, axis=0)
            forces = np.linalg.norm(forces, axis=1)
            delta = np.max(forces)
            if delta > error_min and delta < error_max:
                Train_set.append(atoms)
                
        if len(Train_set) > n:
            Train_set = random.sample(Train_set, n)
        return Train_set

    def calculate_forces_for_symbol(self, calc, symbol):
        ref_force = []
        model_force = []
        for atoms in self.frames:
            atoms.calc = calc
            for atom in atoms:
                if atom.symbol == symbol:
                    ref_force.append(atoms.info['forces'][atom.index])
                    model_force.append(atoms.get_forces()[atom.index])
        return np.array([ref_force, model_force])
    
    def calc(self, calc):
        frames = []
        for atoms in self.frames:
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            atoms_copy.info['energy'] = atoms_copy.get_potential_energy()
            atoms_copy.info['forces'] = atoms_copy.get_forces()
            atoms_copy.info['stress'] = atoms_copy.get_stress()
            frames.append(atoms_copy)
        return frames

    def deform(self, scale = np.arange(0.95, 1.06, 0.05)):
        frames = []
        for atoms in self.frames:
            for s in scale:
                atoms_copy = atoms.copy()
                atoms_copy.set_cell(atoms.get_cell() * s, scale_atoms=True)
                frames.append(atoms_copy)
        return frames
    
    def random_strain(self, ratio = 0.04):
        frames = []
        for atoms in self.frames:
            atoms_copy = atoms.copy()
            strain_matrix = np.eye(3) + 2 * ratio * (np.random.random((3,3)) - 0.5)
            new_cell = np.dot(atoms.get_cell(), strain_matrix)
            atoms_copy.set_cell(new_cell, scale_atoms=True)
            frames.append(atoms_copy)
        return frames
    
    def random_displacement(self, max_displacement = 0.4):
        frames = []
        for atoms in self.frames:
            atoms_copy = atoms.copy()
            atoms_copy.positions += np.random.uniform(-max_displacement, max_displacement, atoms_copy.positions.shape)
            frames.append(atoms_copy)
        return frames
    
    def get_vacancies(self, calc = None, num = [1,3,6]):
        frames = []
        for atoms in self.frames:
            for n in num:
                atoms_copy = atoms.copy()
                Morph(atoms_copy).create_vacancies(n)
                if calc is not None:
                    atoms_copy.calc = calc
                    relax_structure(atoms_copy)
                frames.append(atoms_copy)
        return frames
    
    def get_sias(self, calc = None, vectors = [(1,1,1),(1,0,0),(1,1,0)]):
        frames = []
        for atoms in self.frames:
            for v in vectors:
                atoms_copy = atoms.copy()
                i = random.randint(0, len(atoms_copy) - 1)
                Morph(atoms_copy).create_self_interstitial_atom(vector=v, index=i)
                if calc is not None:
                    atoms_copy.calc = calc
                    relax_structure(atoms_copy, fmax=0.05)
                frames.append(atoms_copy)
        return frames

    def subtract_isolated_atom_energy(self, isolated_atom_energy = {}):
        for atoms in self.frames:
            for atom in atoms:
                atoms.info['energy'] -= isolated_atom_energy[atom.symbol]

    def shuffle_symbols(self):
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            random.shuffle(s)
            atoms.set_chemical_symbols(s)

    def dump(self, filename):
        for atoms in self.frames:
            dump_xyz(filename, atoms)
