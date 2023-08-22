#from Pynep:https://github.com/bigd4/PyNEP/tree/master/pynep
from pynep.calculate import NEP 
from pynep.io import load_nep 
from wizard.io import plot_e, plot_f
import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker 

def main():
    frames = load_nep('train.xyz', ftype = "exyz") 
    print(len(frames))
    calc = NEP('nep.txt')
    e_1, e_2, f_1, f_2 = [], [], [], []
    for atoms in frames:
        atoms.calc = calc
        e_1.append(atoms.get_potential_energy() / len(atoms))
        e_2.append(atoms.info['energy'] / len(atoms))
        f_1.append(atoms.get_forces())
        f_2.append(atoms.info['forces'])
    e_1 = np.array(e_1)
    e_2 = np.array(e_2)
    f_1 = np.concatenate(f_1)
    f_2 = np.concatenate(f_2)
    plot_e(e_2, e_1)
    plot_f(f_2, f_1,[-5,10])
    e_rmse = np.sqrt(np.mean((e_1-e_2)**2)) 
    f_rmse = np.sqrt(np.mean((f_1-f_2)**2))
    print(e_rmse)
    print(f_rmse)
    
if  __name__ == "__main__":   
    main()
