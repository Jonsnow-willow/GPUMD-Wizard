#from Pynep:https://github.com/bigd4/PyNEP/tree/master/pynep
from pynep.calculate import NEP 
from pynep.io import load_nep 
import numpy as np 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker 

def plot_e(ed, er):
    fig = plt.figure()
    plt.title("NEP energy vs DFT energy", fontsize=16)
    ed = ed - np.mean(ed)
    er = er - np.mean(er)
    ax = plt.gca()
    ax.set_aspect(1)
    xmajorLocator = ticker.MaxNLocator(5)
    ymajorLocator = ticker.MaxNLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    
    ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
    xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    
    ax.set_xlabel('DFT energy (eV/atom)', fontsize=14)
    ax.set_ylabel('NEP energy (eV/atom)', fontsize=14)
    
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)    
    ax.tick_params(labelsize=16)

    
    plt.plot([np.min(ed), np.max(ed)], [np.min(ed), np.max(ed)],
            color='black',linewidth=3,linestyle='--',)
    plt.scatter(ed, er, zorder=200)
    
    m1 = min(np.min(ed), np.min(er))
    m2 = max(np.max(ed), np.max(er))
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((ed-er)**2))
    plt.text(np.min(ed) * 0.85 + np.max(ed) * 0.15, 
             np.min(er) * 0.15 + np.max(ed) * 0.85,
             "RMSE: {:.3f} eV/atom".format(rmse), fontsize=14)
    plt.savefig('e.png')
    return fig

def plot_f(fd, fr, lim=None):
    fig = plt.figure()
    ax = plt.gca()
    plt.title("NEP forces vs DFT forces", fontsize=16)
    ax.set_aspect(1)
    xmajorLocator = ticker.MaxNLocator(5)
    ymajorLocator = ticker.MaxNLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    
    ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
    xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    
    ax.set_xlabel('DFT forces (eV/A)', fontsize=14)
    ax.set_ylabel('NEP forces (eV/A)', fontsize=14)
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    ax.tick_params(labelsize=14)

    ax.set_xlim(np.min(fd), np.max(fd))
    ax.set_ylim(np.min(fr), np.max(fr))

    plt.plot([np.min(fd), np.max(fd)], [np.min(fd), np.max(fd)],
            color='black',linewidth=2,linestyle='--')
    plt.scatter(fd.reshape(-1), fr.reshape(-1), s=2)

    m1 = min(np.min(fd), np.min(fr))
    m2 = max(np.max(fd), np.max(fr))
    if lim is not None:
        m1 = lim[0]
        m2 = lim[1]
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((fd-fr)**2))
    plt.text(np.min(fd) * 0.85 + np.max(fd) * 0.15, 
             np.min(fr) * 0.15 + np.max(fr) * 0.85,
             "RMSE: {:.3f} eV/A".format(rmse), fontsize=14)
    plt.savefig('f.png')
    return fig

def main():
    #test_set = sys.argv[1]  # 获取命令行中传递的第一个参数
    frames = load_nep('train_set/Fe/db_Fe.xyz', ftype = "exyz") 
    print(len(frames))
    calc = NEP('potential/Fe.txt')
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
