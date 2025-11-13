from matplotlib import pyplot as plt
from ase.data import chemical_symbols, reference_states
import json

with open('data.json', 'r') as f:
     data = json.load(f)

crystal_structures = {s: r['symmetry'] if r is not None else None
                      for s, r in zip(chemical_symbols, reference_states)}
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 10
fig, axes = plt.subplots(
    figsize=(10, 3),
    dpi=200,
    ncols=3,
    sharex = False,
    sharey=False
)

for k, (species, values) in enumerate(data.items()):
    structure = crystal_structures[species]
    color = f'C{k}'
    if structure == 'fcc':
        axes[0].plot(values['volume'], values['NEP'], 'o-', color=color, label=species, fillstyle='none')
        axes[0].plot(values['volume'], values['DFT'], '*', color=color)
        axes[0].legend(loc = 'upper right')
        axes[0].set_xlabel('Volume (Å$^3$/atom)')
        axes[0].set_ylabel('Formation energy (eV/atom)')
    if structure == 'bcc':
        axes[1].plot(values['volume'], values['NEP'], 'o-', color=color, label=species, fillstyle='none')
        axes[1].plot(values['volume'], values['DFT'], '*', color=color)
        axes[1].legend(loc = 'upper right')
        axes[1].set_xlabel('Volume (Å$^3$/atom)')
        axes[0].set_ylabel('Formation energy (eV/atom)')
    if structure == 'hcp':
        axes[2].plot(values['volume'], values['NEP'], 'o-', color=color, label=species, fillstyle='none')
        axes[2].plot(values['volume'], values['DFT'], '*', color=color)
        axes[2].legend(loc = 'upper right')
        axes[2].set_xlabel('Volume (Å$^3$/atom)')
        axes[0].set_ylabel('Formation energy (eV/atom)')

fig.tight_layout()
fig.align_ylabels(axes)
fig.savefig('eos_curve.png')
plt.close(fig)
