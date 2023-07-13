from wizard.atoms import SymbolInfo, MaterialCalculator
from pynep.calculate import NEP

def main():
    symbol_infos = [
    SymbolInfo('Ag', 'fcc', 4.146),
    SymbolInfo('Al', 'fcc', 4.042),
    SymbolInfo('Au', 'fcc', 4.159),
    SymbolInfo('Cu', 'fcc', 3.631),
    SymbolInfo('Ni', 'fcc', 3.509),
    SymbolInfo('Pb', 'fcc', 5.038),
    SymbolInfo('Pd', 'fcc', 3.939),
    SymbolInfo('Pt', 'fcc', 3.967),
    SymbolInfo('Cr', 'bcc', 2.845),
    SymbolInfo('Fe', 'bcc', 2.759),
    SymbolInfo('Mo', 'bcc', 3.164),
    SymbolInfo('Ta', 'bcc', 3.319),
    SymbolInfo('V', 'bcc', 2.997),
    SymbolInfo('W', 'bcc', 3.185),
    SymbolInfo('Co', 'hcp', 2.256, 6.180),
    SymbolInfo('Mg', 'hcp', 3.195, 5.186),
    SymbolInfo('Ti', 'hcp', 2.931, 4.651),
    SymbolInfo('Zr', 'hcp', 3.240, 5.157)
    ]
    millers = [(1,1,0),(0,0,1),(1,1,1),(1,1,2)]
    sia_vectors = [(1/2,1/2,1/2),(1,0,0),(1,1,0)]
    nths = [1,2,3,4,5]
    calc = NEP('nep.txt')
    for symbol_info in symbol_infos:
        atoms = symbol_info.create_bulk_atoms()
        material_calculator = MaterialCalculator(atoms, calc, symbol_info.symbol, symbol_info.structure)
        material_calculator.lattice_constant()
        material_calculator.eos_curve()
        material_calculator.elastic_constant()
        material_calculator.phonon_dispersion()
        material_calculator.formation_energy_vacancy()
        material_calculator.migration_energy_vacancy()
        for nth in nths:
            material_calculator.formation_energy_divacancies(nth)
        for miller in millers:
            material_calculator.formation_energy_surface(miller)
        material_calculator.stacking_fault(a = (-1,1,1), b = (1,1,0), distance = symbol_info.lattice_constant[0] / 2)
        material_calculator.stacking_fault(a = (1,1,-1), b = (1,1,2), distance = symbol_info.lattice_constant[0] / 2)
        material_calculator.pure_bcc_metal_screw_dipole_move()
        material_calculator.pure_bcc_metal_screw_one_move()
        for vector in sia_vectors:
            material_calculator.formation_energy_sia(vector)
        material_calculator.formation_energy_interstitial_atom(symbol_info.symbol,[0,0,1/2],'octahedral')
        material_calculator.formation_energy_interstitial_atom(symbol_info.symbol,[1/4,0,1/2],'tetrahedral')
        
if __name__ == "__main__":
    main()

