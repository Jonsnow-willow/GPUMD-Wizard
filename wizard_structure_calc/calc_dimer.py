import numpy as np
from pynep.calculate import NEP
from wizard.io import read_xyz
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20, 'lines.linewidth': 2, 'lines.markersize': 5})

def dump_dimer(symbol1, symbol2, distance):
    f = open('dimer.xyz', 'a')
    Out_string = ""
    Out_string += "2\n"
    Out_string += "pbc=\"T T T\" "
    Out_string += "Lattice=\"30.0 0.0 0.0 0.0 18.0 0.0 0.0 0.0 17.0\" "
    Out_string += "Properties=species:S:1:pos:R:3\n"
    Out_string += '{:2} 0 0 0\n'.format(symbol1)
    Out_string += '{:2} {} 0 0\n'.format(symbol2, distance)
    f.write(Out_string)
    f.close()

start = 1.0
end = 1.5
step = 0.05
sequence = np.arange(start, end + step, step)
symbols = ['Fe', 'H']
labels = []
for i, symbol1 in enumerate(symbols):
    for symbol2 in symbols[i:]:
        labels.append(symbol1 + '-'+ symbol2)
        for distance in sequence:
            dump_dimer(symbol1=symbol1, symbol2=symbol2, distance=distance)
    
calcs = [NEP('nep.txt'), NEP('test.txt')]
frames = read_xyz('dimer.xyz')
energies = []
for calc in calcs:
    calc_energies = []
    for atoms in frames:
        atoms.calc = calc
        energy = atoms.get_potential_energy() / 2
        calc_energies.append(energy)
    energies.append(calc_energies)

energies_split = []
for i in range(0, len(energies[0]), len(sequence)):
    energies_split.append([energies[0][i:i+len(sequence)], energies[1][i:i+len(sequence)]])
energies_split = np.array(energies_split)

fig, axs = plt.subplots(len(labels), 1, figsize=(10, 20))
for i, label in enumerate(labels):
    axs[i].plot(sequence, energies_split[i][0], label='same_cutoff')
    axs[i].plot(sequence, energies_split[i][1], label='flexible_cutoff')
    axs[i].scatter(sequence, energies_split[i][0], color='blue')
    axs[i].scatter(sequence, energies_split[i][1], color='orange')
    axs[i].set_xlabel('Distance (Å)')
    axs[i].set_ylabel('Energy (eV)')
    axs[i].set_title(label)
    axs[i].legend()

plt.subplots_adjust(hspace=0.5)

plt.savefig('dimer.png')