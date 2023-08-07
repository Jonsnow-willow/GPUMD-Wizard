from pynep.io import load_nep
from pynep.calculate import NEP
from ase.calculators.lammpslib import LAMMPSlib
import numpy as np
import matplotlib.pyplot as plt

calc_unep = NEP('nep.txt')
cmds = ["pair_style eam/alloy",
        "pair_coeff * * MoTaVW.eam.alloy Mo Ta V W" ]
calc_lmp = LAMMPSlib(lmpcmds=cmds, log_file='log.txt', keep_alive=True)
frames = load_nep('dump.xyz', ftype="exyz")
dft, unep, zhou = [], [], []
for atoms in frames:
    dft.append(atoms.info['forces'])
for atoms in frames:
    atoms.calc = calc_unep
    unep.append(atoms.get_forces())
for atoms in frames:
    atoms.calc = calc_lmp
    zhou.append(atoms.get_forces())

data = np.column_stack((np.concatenate(dft), np.concatenate(unep), np.concatenate(zhou)))
np.savetxt('force_column.txt', data, fmt='%.6f', delimiter='\t')

x = data[:, 0:3]
y1 = data[:, 3:6]
y2 = data[:, 6:9]

plt.scatter(x.flatten(), y2.flatten(), label='Zhou')
plt.scatter(x.flatten(), y1.flatten(), label='UNEP')
plt.plot([x.min(), x.max()], [x.min(), x.max()], color='gray', linestyle='--')

plt.xlabel('DFT')
plt.ylabel('Force')
plt.legend()

plt.axis('equal')
plt.xlim([x.min(), x.max()])
plt.ylim([x.min(), x.max()])

plt.show()