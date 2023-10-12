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

    def select_by_symbol_all(self, symbols = []):
        select_set = []
        split_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            if all(i in symbols for i in s):
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set
    
    def select_by_symbol_any(self, symbols = []):
        select_set = []
        split_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            if any(i in symbols for i in s):
                select_set.append(atoms)
            else:
                split_set.append(atoms)
        return select_set, split_set
    
    def select_by_symbol_num(self, num):
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
    
    def select_by_error(self, error_min, error_max = 100, n = 1000000):
        select_set = []
        for atoms in self.frames:
            f_1 = np.concatenate(atoms.get_forces())
            f_2 = np.concatenate(atoms.info['forces'])
            diff = f_1 - f_2
            if np.any((diff > error_min) & (diff < error_max)):
                select_set.append(atoms)
        if len(select_set) > n:
            select_set = random.sample(select_set, n)
        return select_set
    
    def select_by_force(self, force_min, force_max):
        select_set = []
        for atoms in self.frames:
            f = np.concatenate(atoms.info['forces'])
            if np.all((f > force_min) & (f < force_max)):
                select_set.append(atoms)
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
