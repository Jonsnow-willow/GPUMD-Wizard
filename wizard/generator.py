from wizard.atoms import SymbolInfo
import itertools

class Generator:
    def __init__(self, elements = [], symbols= None, structures = ['bcc', 'fcc', 'hcp'], adjust_ratios = None):
        self.symbol_infos = []
        self.elements = elements
        self.structures = structures
        
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
            for structure in structures:
                if structure == 'hcp':
                    self.symbol_infos.append(SymbolInfo(formula, structure, 3.5, 4.5))
                else:
                    self.symbol_infos.append(SymbolInfo(formula, structure, 3.5))

    def get_symbol_infos(self):
        return self.symbol_infos
                
            


        
            

        
        




