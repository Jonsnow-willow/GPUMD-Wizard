from wizard.structure.atoms import AlloyInfo
from calorine.tools import relax_structure
import itertools

class Generator:
    def __init__(self, elements = [], symbols= None, lattice_types = ['bcc', 'fcc', 'hcp'], adjust_ratios = None):
        self.alloy_infos = []        
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
                    self.alloy_infos.append(AlloyInfo(formula, lattice_type, 3.5, 4.5))
                else:
                    self.alloy_infos.append(AlloyInfo(formula, lattice_type, 3.5))

    def __str__(self):
        for alloy_info in self.alloy_infos:
            print(alloy_info)
        return ''

    def get_alloy_infos(self):
        return self.alloy_infos
    
    def get_bulk_structures(self, calc = None, supercell = {'bcc': (3,3,3), 'fcc': (3,3,3), 'hcp': (3,3,3)}):
        frames = []
        for alloy_info in self.alloy_infos:
            atoms = alloy_info.create_bulk_atoms(supercell[alloy_info.lattice_type])
            if calc is not None:
                atoms.calc = calc
                relax_structure(atoms)
            frames.append(atoms)
        return frames
       
        
                
            


        
            

        
        




