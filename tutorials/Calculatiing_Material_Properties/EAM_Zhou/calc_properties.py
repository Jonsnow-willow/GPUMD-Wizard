from wizard.atoms import SymbolInfo
from wizard.calculator import MaterialCalculator
from ase.calculators.lammpslib import LAMMPSlib

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
    SymbolInfo('Mo', 'bcc', 3.164),
    SymbolInfo('Ta', 'bcc', 3.319),
    SymbolInfo('V', 'bcc', 2.997),
    SymbolInfo('W', 'bcc', 3.185),
    SymbolInfo('Mg', 'hcp', 3.195, 5.186),
    SymbolInfo('Ti', 'hcp', 2.931, 4.651),
    SymbolInfo('Zr', 'hcp', 3.240, 5.157)
    ]
    millers = [(1,1,0),(0,0,1),(1,1,1),(1,1,2)]
    vectors = [(1,0,0),(1,1,0),(1,1,1)]
    for symbol_info in symbol_infos:
        # calculate basic properties
        atoms = symbol_info.create_bulk_atoms((1,1,1))
        cmds = ["pair_style eam/alloy",
                "pair_coeff * * potentials/" + symbol_info.formula + ".eam.alloy " + symbol_info.formula]
        calc = LAMMPSlib(lmpcmds=cmds, log_file='log.' + symbol_info.formula, keep_alive=True)
        material_calculator = MaterialCalculator(atoms, calc, symbol_info)
        material_calculator.lattice_constant()
        material_calculator.elastic_constant()
        material_calculator.eos_curve()
        for miller in millers:
            material_calculator.formation_energy_surface(miller)

        # calculate phonon dispersion
        atoms = symbol_info.create_bulk_atoms((3,3,3))
        material_calculator = MaterialCalculator(atoms, calc, symbol_info)
        material_calculator.phonon_dispersion()

        # calculate point defect properties
        atoms = symbol_info.create_bulk_atoms((3,4,5))
        material_calculator = MaterialCalculator(atoms, calc, symbol_info, fmax=0.02)
        material_calculator.formation_energy_vacancy()
        for v in vectors:
            material_calculator.formation_energy_sia(v)

          
if __name__ == "__main__":
    main()

