from wizard.io import dump_xyz
import numpy as np 
import random 

class MultiMol(): 
    
    def __init__(self, frames):         
        self.frames = frames

    def select_binary(self):
        select_set = []
        split_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            unique_elements = set(s)
            if len(unique_elements) < 3:
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set

    def select_multi(self):
        select_set = []
        split_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            unique_elements = set(s)
            if len(unique_elements) > 2:
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set
    
    def select_by_size(self, num):
        select_set = []
        split_set = []
        for atoms in self.frames:
            if len(atoms) == num:
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set

    def select_by_symbols(self, symbols_list = []):
        select_set = []
        split_set = []
        for atoms in self.frames:
            symbols = atoms.get_chemical_symbols()
            if all(symbol in symbols_list for symbol in symbols):
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set

    def select_by_num_of_symbols(self, num):
        select_set = []
        split_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            unique_elements = set(s)
            if len(unique_elements) == num:
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set
    
    def select_by_error(self, error_min, error_max = 100):
        select_set = []
        split_set = []
        for atoms in self.frames:
            f_1 = np.concatenate(atoms.get_forces())
            f_2 = np.concatenate(atoms.info['forces'])
            diff = abs(f_1 - f_2)
            if np.any((diff > error_min) & (diff < error_max)):
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set
    
    def select_by_force(self, force_min, force_max):
        select_set = []
        split_set = []
        for atoms in self.frames:
            f = np.concatenate(atoms.info['forces'])
            if np.all((f > force_min) & (f < force_max)):
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set
    
    def select_random(self, num):
        select_set = []
        split_set = []
        select_set = random.sample(self.frames, num)
        split_set = [i for i in self.frames if i not in select_set]
        return select_set, split_set
    
    def subtract_frames(self, frames):
        select_set = [i for i in self.frames if i not in frames]
        return select_set
    
    def subtract_isolated_atom_energy(self, isolated_atom_energy = {}):
        for atoms in self.frames:
            for atom in atoms:
                atoms.info['energy'] -= isolated_atom_energy[atom.symbol]
    
    def cal_force_by_symbol(self, symbol):
        dft_force = []
        calc_force = []
        for atoms in self.frames:
            for atom in atoms:
                if atom.symbol == symbol:
                    dft_force.append(atoms.info['forces'][atom.index])
                    calc_force.append(atoms.get_forces()[atom.index])
        return np.array([dft_force, calc_force])

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
    
    def shuffle_symbols(self):
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            random.shuffle(s)
            atoms.set_chemical_symbols(s)

    def dump_sequence(self, filename):
        for atoms in self.frames:
            dump_xyz(filename, atoms)