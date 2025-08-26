from wizard.atoms import SymbolInfo
from wizard.io import relax
import itertools

class Generator:
    def __init__(self, elements = [], symbols= None, lattice_types = ['bcc', 'fcc', 'hcp'], adjust_ratios = None):
        self.symbol_infos = []        
        combinations = list(itertools.combinations(elements, 2))
        if symbols is not None:
            symbols = list(set(symbols) - set(elements))
            if symbols:
                combinations += [(a, b) for a in symbols for b in elements]
        formulas = []
        if adjust_ratios is not None:
            ratio = int(adjust_ratios)
            for a, b in combinations:
                for i in range(1, ratio):
                    formulas.append(f"{a}{i}{b}{ratio-i}")
        else:
            formulas = [''.join(combination) for combination in combinations]
        for element in elements:
            formulas.append(element)

        for formula in formulas:
            for lattice_type in lattice_types:
                if lattice_type == 'hcp':
                    self.symbol_infos.append(SymbolInfo(formula, lattice_type, 3.5, 4.5))
                else:
                    self.symbol_infos.append(SymbolInfo(formula, lattice_type, 3.5))

    def __str__(self):
        for symbol_info in self.symbol_infos:
            print(symbol_info)
        return ''

    def get_symbol_infos(self):
        return self.symbol_infos
    
    def get_bulk_structures(self, calc = None, supercell = {'bcc': (3,3,3), 'fcc': (3,3,3), 'hcp': (3,3,3)}):
        frames = []
        for symbol_info in self.symbol_infos:
            atoms = symbol_info.create_bulk_atoms(supercell[symbol_info.lattice_type])
            if calc is not None:
                atoms.calc = calc
                relax(atoms, steps=100)
            frames.append(atoms)
        return frames
       
        
                
            


        
            

        
        




