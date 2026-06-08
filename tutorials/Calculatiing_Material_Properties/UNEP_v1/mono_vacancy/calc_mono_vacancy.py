from wizard.model.atoms import AlloyInfo
from wizard.calc.calculator import MaterialCalculator
from calorine.calculators import CPUNEP
import matplotlib.pyplot as plt

data = {'DFT':{}, 'NEP':{}}

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

for alloy_info in alloy_infos:
    atoms = alloy_info.create_bulk_atoms((3,4,5))
    calc = CPUNEP('../potentials/4-4-80/nep.txt')
    material_calculator = MaterialCalculator(atoms, calc)
    e = material_calculator.formation_energy_vacancy()
    data['NEP'][alloy_info.formula] = e

symbols = ['Ag', 'Al', 'Au', 'Cu', 'Ni', 'Pb', 'Pd', 'Pt', 'Cr', 'Mo', 'Ta', 'V', 'W', 'Mg', 'Ti', 'Zr']

data['DFT'] = {"Ag": 0.96, "Al": 0.55, "Au": 0.51, "Cu": 1.12, "Ni": 1.41, "Pb": 0.39, "Pd": 1.23, "Pt": 0.68, "Cr": 2.54, "Mo": 2.77, "Ta": 2.82, "V": 2.28, "W": 3.34, "Mg": 0.85, "Ti": 2.00, "Zr": 2.00}
fig, ax = plt.subplots(figsize=(14, 3))  
plt.rcParams.update({'font.size': 15})

ax.plot(symbols, [data['NEP'][symbol] for symbol in symbols], marker='o', label='UNEP_v1', fillstyle='none')
ax.plot(symbols, [data['DFT'][symbol] for symbol in symbols], marker='*', label='DFT', markersize=8)
ax.set_ylabel('Formation Energy (eV)')
ax.legend()
fig.patch.set_facecolor('white')
fig.savefig('mono_vacancy.png')
plt.close(fig)
