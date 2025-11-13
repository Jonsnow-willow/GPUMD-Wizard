from matplotlib import pyplot as plt
from ase.data import chemical_symbols, reference_states
from pandas import DataFrame
import json

with open('data.json', 'r') as f:
    data = json.load(f)

df_surface_energies = DataFrame.from_dict(data)

crystal_structures = {s: r['symmetry'] if r is not None else None
                      for s, r in zip(chemical_symbols, reference_states)}


plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 6
fig, axes = plt.subplots(
    figsize=(6, 8),
    dpi=200,
    nrows=3,
    sharex = False,
    sharey=False
)

for k, (species, df) in enumerate(df_surface_energies.groupby('species')):
    structure = crystal_structures[species]
    if structure == 'hcp':
        continue
    irow = 0 if structure == 'fcc' else 1
    ax = axes[irow]
    df.reset_index(inplace=True)
    df.sort_values('hkl', inplace=True)  
    df.reset_index(drop=True, inplace=True)
    color = f'C{k}'
    ax.plot(df.gamma_NEP, 'o-', color=color, label=species, fillstyle='none', markersize=6, linewidth=1)
    ax.plot(df.gamma_DFT, '*', color=color, markersize=6, linewidth=1)
    ax.legend(loc='upper right')
    ax.set_xticks(range(len(df.hkl)))
    ax.set_xticklabels(df.hkl, rotation=60)
    ax.set_xlabel('Surface orientation')
    ax.set_ylabel('Surface energy (J/m^2)')

for k, (species, df) in enumerate(df_surface_energies[df_surface_energies['species'].isin(['Mg', 'Zr', 'Ti'])].groupby('species')):
    df.reset_index(inplace=True)
    df.sort_values('hkl', inplace=True)  
    df.reset_index(drop=True, inplace=True)
    color = f'C{k}'
    axes[2].plot(df.gamma_NEP, 'o-', color=color, label=species, fillstyle='none', markersize=6, linewidth=1)
    axes[2].plot(df.gamma_DFT, '*', color=color, markersize=6, linewidth=1)
    axes[2].legend(loc='upper right')
    axes[2].set_xticks(range(len(df.hkl)))
    axes[2].set_xticklabels(df.hkl, rotation=60)
    axes[2].set_xlabel('Surface orientation')
    axes[2].set_ylabel('Surface energy (J/m^2)')

fig.tight_layout()
fig.align_ylabels(axes)
fig.savefig('surface_energies.png')
plt.close(fig)