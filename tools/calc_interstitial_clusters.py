from wizard.atoms import SymbolInfo, Morph
from wizard.io import dump_xyz, mkdir_relax, read_xyz, run_gpumd
import numpy as np
import os

def relax(atoms):
    mkdir_relax(atoms, run_in = ['potential ../nep.txt', 'ensemble nve', 'time_step 0',
                                     'minimize fire 1.0e-4 1000','dump_exyz 1 0 0','run 1'])
    run_gpumd('relax')
    atoms = read_xyz('relax/dump.xyz')[-1]
    os.system('rm -rf relax')
       
def formation_energy_sias_cluster(atoms, energy, symbol, burger, r = 5, thickness = 2, vector = (1, 0, 0)):
    initial_num = len(atoms)
    defects = atoms.copy() 
    center = defects.get_center_of_mass()
    for atom in defects:
        vector = atom.position - center
        proj = abs(vector @ vector) / np.linalg.norm(vector)
        R = np.sqrt(max(np.dot(vector, vector) - np.dot(proj, proj), 0))
        if  R < r and proj < thickness:
            Morph(atoms).create_self_interstitial_atom(vector=vector, index = atom.index)
    relax(defects)
    formation_energy = defects.info[energy] - energy * len(defects) / initial_num
    cluster_num = len(defects) - initial_num
    
    dump_xyz('loop.xyz', atoms, comment=f' config_type = {symbol} {burger} {cluster_num} SIAs Cluster')
    with open('formation_energy_loops.out', 'a') as f:
        print(f' {symbol:<7}{burger} {cluster_num} sias Formation_Energy_Sias_Cluster: {formation_energy:.4} eV', file=f)


def main():
    symbol_info = SymbolInfo('W', 'bcc', 3.185)
    atoms = symbol_info.create_bulk_atoms()
    relax(atoms)
    e = atoms.info['energy']
    for r in range(4,15):
        formation_energy_sias_cluster(atoms, e, 'W', '1/2<111>', r = r, vector=(0.7,-0.7,0))


if __name__ == "__main__":
    main()

    