from ase.data import chemical_symbols, reference_states
from wizard.molecular_dynamics import MolecularDynamics
from wizard.atoms import Morph, SymbolInfo
from wizard.frames import MultiMol
from calorine.calculators import CPUNEP
from ase.build import bulk
import itertools

class Generator:
    def __init__(self, elements = [], symbols= None, structures = ['bcc', 'fcc', 'hcp'], adjust_ratios = False):
        self.symbol_infos = []
        self.elements = elements
        self.structures = structures
        
        combinations = list(itertools.combinations(elements, 2))
        if symbols is not None:
            symbols = list(set(symbols) - set(elements))
            if symbols:
                combinations += [(a, b) for a in symbols for b in elements]
        formulas = []
        if (adjust_ratios):
            for a, b in combinations:
                for i in range(1, 5):
                    formulas.append(f"{a}{i}{b}{5-i}")
        else:
            formulas = [''.join(combination) for combination in combinations]
        for element in elements:
            formulas.append(element)

        for formula in formulas:
            for structure in structures:
                if structure == 'hcp':
                    self.symbol_infos.append(SymbolInfo(formula, structure, 3.5, 4.5))
                else:
                    self.symbol_infos.append(SymbolInfo(formula, structure, 3.5))

    def get_symbol_infos(self):
        return self.symbol_infos
                
            


        
            

        
        




