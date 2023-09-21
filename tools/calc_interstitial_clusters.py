from wizard.atoms import SymbolInfo, Morph
from wizard.io import dump_xyz, mkdir_relax, read_xyz, run_gpumd
import numpy as np
import os

def relax(atoms):
    mkdir_relax(atoms, run_in = ['potential ../nep.txt', 'ensemble nve', 'time_step 0',
                                     'minimize fire 1.0e-4 1000','dump_exyz 1 0 0','run 1'])
    run_gpumd('relax')
    atoms_relaxed = read_xyz('relax/dump.xyz')[-1]
    os.system('rm -rf relax')
    return atoms_relaxed
       
def formation_energy_sias_cluster(atoms, info, Rcut = 5, thickness = 2, burger = (1, 0, 0)):
    cluster = atoms.copy() 
    center = cluster.get_center_of_mass()
    for atom in cluster:
        vector = atom.position - center
        proj = abs(vector @ burger) / np.linalg.norm(burger)
        R = np.sqrt(max(np.dot(vector, vector) - np.dot(proj, proj), 0))
        if  R < Rcut and proj < thickness:
            Morph(cluster).create_self_interstitial_atom(burger, index = atom.index)
    cluster_relaxed = relax(cluster)
    formation_energy = cluster_relaxed.info['energy'] - atoms.info['energy'] * len(cluster) / len(atoms)
    cluster_num = len(cluster) - len(atoms)
    burger = tuple(round(x) for x in burger)

    dump_xyz('SIA_cluster.xyz', atoms, comment=f' config_type = {info}{cluster_num}_SIAs_Cluster')
    with open('SIA_cluster.out', 'a') as f:
        print(f' {info}{cluster_num}_sias_Formation_Energy_Sias_Cluster: {formation_energy:.4} eV', file=f)
    return formation_energy

def main():
    symbol_info = SymbolInfo('W', 'bcc', 3.185)
    atoms = symbol_info.create_bulk_atoms() * (50, 50, 50)
    atoms_relaxed = relax(atoms)
    for r in range(4,15):
        formation_energy_sias_cluster(atoms_relaxed, 'W_1/2<111>_', Rcut = r, thickness= 2.1, burger=(0.7,-0.7,0))

if __name__ == "__main__":
    main()

    