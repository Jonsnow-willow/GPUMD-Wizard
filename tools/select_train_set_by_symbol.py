#from Pynep:https://github.com/bigd4/PyNEP/tree/master/pynep
from wizard.frames import MultiMol
from pynep.calculate import NEP 
from pynep.io import load_nep 
from wizard.io import plot_e, plot_f
import numpy as np 
import matplotlib 
matplotlib.use('Agg') 

def main():
    symbols = ['Ag', 'Al', 'Au', 'Co', 'Cr', 'Cu', 'Fe', 'Mg', 'Mo', 'Ni', 'Pb', 'Pd', 'Pt', 'Ta', 'Ti', 'V', 'W', 'Zr']
    for symbol in symbols:
        frames = load_nep('test.xyz', ftype= "exyz")
        mol = MultiMol(frames)
        structures, _ = mol.select_contain_all_symbols([symbol])
        calc = NEP('nep.txt')
        e_1, e_2, f_1, f_2 = [], [], [], []
        for atoms in structures:
            atoms.calc = calc
            e_1.append(atoms.get_potential_energy() / len(atoms))
            e_2.append(atoms.info['energy'] / len(atoms))
            f_1.append(atoms.get_forces())
            f_2.append(atoms.info['forces'])
        e_1 = np.array(e_1)
        e_2 = np.array(e_2)
        f_1 = np.concatenate(f_1)
        f_2 = np.concatenate(f_2)
        plot_e(e_2, e_1, symbol)
        plot_f(f_2, f_1, symbol)
        e_rmse = np.sqrt(np.mean((e_1-e_2)**2)) 
        f_rmse = np.sqrt(np.mean((f_1-f_2)**2))
        print(e_rmse)
        print(f_rmse)
    
if  __name__ == "__main__":   
    main()
