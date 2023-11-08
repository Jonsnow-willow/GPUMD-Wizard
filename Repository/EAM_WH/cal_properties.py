from wizard.atoms import SymbolInfo, MaterialCalculator
from ase.calculators.lammpslib import LAMMPSlib

def main():
    symbol_info = SymbolInfo('W', 'bcc', 3.185)
    millers = [(1,1,0),(0,0,1),(1,1,1),(1,1,2)]
    sia_vectors = [(1/2,1/2,1/2),(1,0,0),(1,1,0)]
    nths = [1,2,3,4,5]
    cmds = ["pair_style eam/alloy",
            "pair_coeff * * WH.eam.alloy W"]
    calc = LAMMPSlib(lmpcmds=cmds, log_file='log.' + symbol_info.symbol, keep_alive=True)
    atoms = symbol_info.create_bulk_atoms()
    material_calculator = MaterialCalculator(atoms, calc, symbol_info.symbol, symbol_info.structure)
    material_calculator.lattice_constant()
    material_calculator.elastic_constant()
    material_calculator.eos_curve()
    for vector in sia_vectors:
        material_calculator.formation_energy_sia(vector)
    material_calculator.formation_energy_interstitial_atom('W',[0,0,1/2],'octahedral')
    material_calculator.formation_energy_interstitial_atom('W',[1/4,0,1/2],'tetrahedral')
    material_calculator.formation_energy_vacancy()
    material_calculator.migration_energy_vacancy()
    for nth in nths:
        material_calculator.formation_energy_divacancies(nth)
    for miller in millers:
        material_calculator.formation_energy_surface(miller)
    material_calculator.stacking_fault(a = (1,1,-1), b = (1,-1,0), miller = [1,1,2], distance = 3.185/2)
    material_calculator.stacking_fault(a = (1,1,-1), b = (1,1,2), miller = [1,1,0], distance = 3.185/2)
    material_calculator.pure_bcc_metal_screw_dipole_move()
    material_calculator.pure_bcc_metal_screw_one_move()

if __name__ == "__main__":
    main()

