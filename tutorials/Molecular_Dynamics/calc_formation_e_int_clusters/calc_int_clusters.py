from wizard.atoms import SymbolInfo, Morph
from wizard.io import read_xyz
import numpy as np
       
burger = (1, 0, 0)
Rcut = 10
thickness = 2
atoms = SymbolInfo('W', 'bcc', 3.185).create_bulk_atoms(supercell=(30, 30, 30))
coh_e = -12.597
center = atoms.get_center_of_mass()
for atom in atoms:
    vector = atom.position - center
    proj = abs(vector @ burger) / np.linalg.norm(burger)
    R = np.sqrt(max(np.dot(vector, vector) - np.dot(proj, proj), 0))
    if  R < Rcut and proj < thickness:
        Morph(atoms).create_self_interstitial_atom(burger, index = atom.index)
Morph(atoms).gpumd(dirname='sia_cluster',
                   run_in= ['potential nep.txt', 'ensemble nve', 'time_step 0',
                            'minimize fire 1.0e-4 1000','dump_exyz 1','run 1'],
                   nep_path='../potentials/nep.txt')
frames = read_xyz('sia_cluster/dump.xyz')
atoms = frames[-1]
formation_energy = atoms.info['energy'] -  coh_e * len(atoms)