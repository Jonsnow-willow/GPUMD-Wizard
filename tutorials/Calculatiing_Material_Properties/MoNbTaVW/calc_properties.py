from wizard.model.atoms import AlloyInfo
from wizard.calc.calculator import AlloyCalculator, MaterialCalculator
from calorine.calculators import CPUNEP

def main():
    alloy_infos = [
    AlloyInfo('V',  'bcc', 2.997),
    AlloyInfo('Nb', 'bcc', 3.308),
    AlloyInfo('Mo', 'bcc', 3.163),
    AlloyInfo('Ta', 'bcc', 3.321),
    AlloyInfo('W',  'bcc', 3.185)
    ]
    calc = CPUNEP('potentials/nep.txt')

    for alloy_info in alloy_infos:
        # calculate basic properties
        atoms = alloy_info.create_bulk_atoms((1,1,1))
        material_calculator = MaterialCalculator(atoms, calc)
        material_calculator.lattice_constant()
        material_calculator.elastic_constant()
        material_calculator.eos_curve()

        atoms = alloy_info.create_bulk_atoms((3,3,3))
        material_calculator = MaterialCalculator(atoms, calc)
        material_calculator.phonon_dispersion()
        material_calculator.formation_energy_vacancy()
        material_calculator.migration_energy_vacancy()

    alloy_info = AlloyInfo('VNbMoTaW',  'bcc', 3.195)
    material_calculator = AlloyCalculator(alloy_info, (3,3,3), calc)
    material_calculator.isolate_atom_energy()
    material_calculator.lattice_constant()
    material_calculator.dimer_curve()
    material_calculator.elastic_constant()
    material_calculator.eos_curve()
    material_calculator.formation_energy_vacancy()
    
if __name__ == "__main__":
    main()
