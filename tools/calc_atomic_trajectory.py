from wizard.io import read_xyz
from pynep.calculate import NEP
import numpy as np
import matplotlib.pyplot as plt

calc_unep = NEP('nep.txt')

frames = read_xyz('train.xyz')
dft, unep = [], []
for atoms in frames:
    dft.append(atoms.info['forces'])
for atoms in frames:
    atoms.calc = calc_unep
    unep.append(atoms.get_forces())

data = np.column_stack((np.concatenate(dft), np.concatenate(unep)))
np.savetxt('force_column.txt', data, fmt='%.6f', delimiter='\t')

x = data[:, 0:3]
y = data[:, 3:6]

plt.scatter(x.flatten(), y.flatten(), label='UNEP')
plt.plot([x.min(), x.max()], [x.min(), x.max()], color='gray', linestyle='--')

plt.xlabel('DFT')
plt.ylabel('Force')
plt.legend()

plt.axis('equal')
plt.xlim([x.min(), x.max()])
plt.ylim([x.min(), x.max()])

plt.show()