from ase import Atoms
from calorine.calculators import CPUNEP
import matplotlib.pyplot as plt 

data = {'UNEP':[]}
distances = [1.5 + i * 0.1 for i in range(20)]
symbol1 = 'W'
symbol2 = 'W'
for distance in distances:
    dimmer = Atoms(symbols= [symbol1, symbol2], positions=[(0, 0, 0), (0, 0, distance)], pbc = [True, True, True] ,cell=(20, 30, 40))
    dimmer.calc = CPUNEP('../potentials/4-4-80/nep.txt')
    data['UNEP'].append(dimmer.get_potential_energy() / len(dimmer))

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)    
ax.plot(distances, data['UNEP'], '-o', label = 'UNEP',fillstyle='none', markersize = 8, color='C0') 
ax.set_xlabel('Distance (Å)')
ax.set_ylabel('Energy (eV/atom)')
ax.legend()
fig.savefig('dimer.png')
plt.close(fig)
