from wizard.model.atoms import AlloyInfo
from wizard.calc.calculator import MaterialCalculator
from ase.calculators.lammpslib import LAMMPSlib

def main():
    alloy_infos = [
    AlloyInfo('Ag', 'fcc', 4.146),
    AlloyInfo('Al', 'fcc', 4.042),
    AlloyInfo('Au', 'fcc', 4.159),
    AlloyInfo('Cu', 'fcc', 3.631),
    AlloyInfo('Ni', 'fcc', 3.509),
    AlloyInfo('Pb', 'fcc', 5.038),
    AlloyInfo('Pd', 'fcc', 3.939),
    AlloyInfo('Pt', 'fcc', 3.967),
    AlloyInfo('Cr', 'bcc', 2.845),
    AlloyInfo('Mo', 'bcc', 3.164),
    AlloyInfo('Ta', 'bcc', 3.319),
    AlloyInfo('V', 'bcc', 2.997),
    AlloyInfo('W', 'bcc', 3.185),
    AlloyInfo('Mg', 'hcp', 3.195, 5.186),
    AlloyInfo('Ti', 'hcp', 2.931, 4.651),
    AlloyInfo('Zr', 'hcp', 3.240, 5.157)
    ]
    millers = [(1,1,0),(0,0,1),(1,1,1),(1,1,2)]
    vectors = [(1,0,0),(1,1,0),(1,1,1)]
    for alloy_info in alloy_infos:
        # calculate basic properties
        atoms = alloy_info.create_bulk_atoms((1,1,1))
        cmds = ["pair_style eam/alloy",
                "pair_coeff * * potentials/" + alloy_info.formula + ".eam.alloy " + alloy_info.formula]
        calc = LAMMPSlib(lmpcmds=cmds, log_file='log.' + alloy_info.formula, keep_alive=True)
        material_calculator = MaterialCalculator(atoms, calc)
        material_calculator.lattice_constant()
        material_calculator.elastic_constant()
        material_calculator.eos_curve()
        for miller in millers:
            material_calculator.formation_energy_surface(miller)

        # calculate phonon dispersion
        atoms = alloy_info.create_bulk_atoms((3,3,3))
        material_calculator = MaterialCalculator(atoms, calc)
        material_calculator.phonon_dispersion()

        # calculate point defect properties
        atoms = alloy_info.create_bulk_atoms((3,4,5))
        material_calculator = MaterialCalculator(atoms, calc, fmax=0.02)
        material_calculator.formation_energy_vacancy()
        for v in vectors:
            material_calculator.formation_energy_sia(v)

          
if __name__ == "__main__":
    main()
