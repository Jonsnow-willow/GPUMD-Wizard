from wizard.atoms import SymbolInfo
from wizard.calculator import MaterialCalculator
from calorine.calculators import CPUNEP

def main():
    symbol_infos = [
    SymbolInfo('V',  'bcc', 2.997),
    SymbolInfo('Nb', 'bcc', 3.308),
    SymbolInfo('Mo', 'bcc', 3.163),
    SymbolInfo('Ta', 'bcc', 3.321),
    SymbolInfo('W',  'bcc', 3.185)
    ]
    calc = CPUNEP('potentials/nep.txt')

    for symbol_info in symbol_infos:
        # calculate basic properties
        atoms = symbol_info.create_bulk_atoms((1,1,1))
        material_calculator = MaterialCalculator(atoms, calc, symbol_info)
        material_calculator.lattice_constant()
        material_calculator.elastic_constant()
        material_calculator.eos_curve()

        atoms = symbol_info.create_bulk_atoms((3,3,3))
        material_calculator = MaterialCalculator(atoms, calc, symbol_info)
        material_calculator.phonon_dispersion()
        material_calculator.formation_energy_vacancy()
        material_calculator.migration_energy_vacancy()

    symbol_info = SymbolInfo('VNbMoTaW',  'bcc', 3.195)
    atoms = symbol_info.create_bulk_atoms((3,3,3))
    material_calculator = MaterialCalculator(atoms, calc, symbol_info)
    material_calculator.isolate_atom_energy()
    material_calculator.lattice_constant()
    material_calculator.dimer_curve()
    material_calculator.elastic_constant()
    material_calculator.eos_curve()
    material_calculator.formation_energy_vacancy()
    
if __name__ == "__main__":
    main()