import numpy as np 
import random 

class MultiMol(): 
    
    def __init__(self, frames):         
        self.frames = frames

    def Devide_Train_set_by_symbol(self, symbols):
        Train_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            if all(i in symbols for i in s):
                Train_set.append(atoms)
        return Train_set

    def Devide_Train_set_by_symbol_binary(self):
        Train_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            unique_elements = set(s)
            if len(unique_elements) < 3:
                Train_set.append(atoms)
        return Train_set

    def Devide_Train_set_by_symbol_multi(self):
        Train_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            unique_elements = set(s)
            if len(unique_elements) > 2:
                Train_set.append(atoms)
        return Train_set
    
    def Devide_Train_set_by_symbol_num(self, num):
        Train_set = []
        for atoms in self.frames:
            s = atoms.get_chemical_symbols()
            unique_elements = set(s)
            if len(unique_elements) == num:
                Train_set.append(atoms)
        return Train_set
    
    def Devide_Train_set_by_num(self, num):
        Train_set = []
        for atoms in self.frames:
            if len(atoms) == num:
                Train_set.append(atoms)
        return Train_set
    
    def select_set_by_error(self, error_min, error_max = 100, n = 1000000):
        Train_set = []
        for atoms in self.frames:
            f_1 = np.concatenate(atoms.get_forces())
            f_2 = np.concatenate(atoms.info['forces'])
            diff = f_1 - f_2
            if np.any((diff > error_min) & (diff < error_max)):
                Train_set.append(atoms)
        if len(Train_set) > n:
            Train_set = random.sample(Train_set, n)
        return Train_set
