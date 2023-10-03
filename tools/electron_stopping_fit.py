import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('electron_stopping.txt')

num_points = 20000
e_start = 10     #eV
e_end = 200000   #eV
stopping_energy = np.linspace(e_start, e_end, num_points)

raw_data = data[:, [0,4,2,1,3,5]]
data_arr = np.array(raw_data)
elements = [
    {'energy': data_arr[:, 0], 'friction': data_arr[:,1:], 'fit': []}
]

for e in elements:
    for j in range(len(e['energy']) - 1):
        e['fit'].append(
            np.polyfit(
                e['energy'][j: j + 2],
                e['friction'][j: j + 2],
                1
            )
        )

def get_friction_forces_for_energy(fit, energy, energy_points):
    for i, en in enumerate(energy_points):
        if en >= energy:
            break
    return np.polyval(fit[i - 1], energy)

friction_forces = []
for e in elements:
    friction_forces.append(
        [
            get_friction_forces_for_energy(e['fit'], en, e['energy'])
            for en in stopping_energy
        ]
    )
friction_forces_arr = np.array(friction_forces[0])
friction_forces_arr[0] = np.zeros(5)

with open('electron_stopping_fit.txt', 'w') as file:
    file.write(f"{num_points} {e_start} {e_end}\n")
    for frictions in friction_forces_arr:
        line = ' '.join(['{:.2f}'.format(f) for f in frictions]) + "\n"
        file.write(line)

for i in range(5):
    plt.scatter(data_arr[:,0], data_arr[:,i+1], label=f"y{i+1}", zorder=1)
    plt.plot(stopping_energy, friction_forces_arr[:,i], label=f"y{i+1}", zorder=2)

plt.legend()
plt.xscale('log')
plt.xlim(10,200000)
plt.ylim(0, 110)
plt.savefig('electron_stopping_fit.png')