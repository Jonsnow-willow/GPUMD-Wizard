"""
# ----------------------------------------------------------------------------------
# Note: This script uses the WNEP2 potential and keeps the original numerical
# settings for reproduction.
# ----------------------------------------------------------------------------------
"""

from wizard.model.atoms import AlloyInfo
from wizard.calc.calculator import AlloyCalculator
from calorine.calculators import CPUNEP


def main():
    alloy_info = AlloyInfo('W', 'bcc', 3.185)
    calc = CPUNEP('nep.txt')
    millers = [(1,1,0),(0,0,1),(1,1,1),(1,1,2),(2,1,0),
               (2,2,1),(3,1,1),(3,1,0),(3,2,1),(3,2,0)]
    sia_vectors = [(1,1,1),(1,0,0),(1,1,0)]
    nths = [1,2,3]
    material_calculator = AlloyCalculator(alloy_info, calc, supercell=(3,3,3))
    material_calculator.lattice_constant()
    material_calculator.elastic_constant()
    material_calculator.eos_curve()
    material_calculator.phonon_dispersion()
    material_calculator.formation_energy_vacancy()
    material_calculator.migration_energy_vacancy()
    for nth in nths:
        material_calculator.formation_energy_divacancies(index0=0, index1=nth)
    for miller in millers:
        material_calculator.formation_energy_surface(miller)
    material_calculator.stacking_fault(a = (1,1,-1), b = (1,-1,0), miller = [1,1,2], distance = 3.185/2)
    material_calculator.stacking_fault(a = (1,1,-1), b = (1,1,2), miller = [1,1,0], distance = 3.185/2)
    material_calculator.bcc_metal_screw_dipole_move()
    material_calculator.bcc_metal_screw_one_move()
    for vector in sia_vectors:
        material_calculator.formation_energy_sia(vector)
    material_calculator.formation_energy_interstitial('W', 'oct')
    material_calculator.formation_energy_interstitial('W', 'tet')
        
if __name__ == "__main__":
    main()
