from wizard.model.atoms import AlloyInfo
from wizard.calc.calculator import MaterialCalculator
from calorine.calculators import CPUNEP

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
    AlloyInfo('V',  'bcc', 2.997),
    AlloyInfo('W',  'bcc', 3.185),
    AlloyInfo('Mg', 'hcp', 3.195, 5.186),
    AlloyInfo('Ti', 'hcp', 2.931, 4.651),
    AlloyInfo('Zr', 'hcp', 3.240, 5.157)
    ]
calc = CPUNEP('../potentials/4-4-80/nep.txt')

for alloy_info in alloy_infos:
    atoms = alloy_info.create_bulk_atoms((3,3,3))
    material_calculator = MaterialCalculator(atoms, calc)
    material_calculator.phonon_dispersion()
