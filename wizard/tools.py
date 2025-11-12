from mpl_toolkits.axes_grid1 import ImageGrid    
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
import numpy as np
import os

def plot_e(ed, en, lim=None, symbol=None):
    fig = plt.figure()
    plt.title("NEP energy vs DFT energy", fontsize=16)
    ed = ed - np.mean(ed)
    en = en - np.mean(en)
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
    plt.scatter(ed, en, zorder=200)
    
    m1 = min(np.min(ed), np.min(en))
    m2 = max(np.max(ed), np.max(en))
    if lim is not None:
        m1 = lim[0]
        m2 = lim[1]
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((ed-en)**2))
    plt.text(np.min(ed) * 0.85 + np.max(ed) * 0.15, 
             np.min(en) * 0.15 + np.max(ed) * 0.85,
             "RMSE: {:.3f} eV/atom".format(rmse), fontsize=14)
    if symbol is None:
        plt.savefig('e.png')
    else:
        plt.savefig(f'{symbol}_e.png')
    return fig

def plot_f(fd, fn, lim=None, symbol=None):
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
    ax.set_ylim(np.min(fn), np.max(fn))

    plt.plot([np.min(fd), np.max(fd)], [np.min(fd), np.max(fd)],
            color='black',linewidth=2,linestyle='--')
    plt.scatter(fd.reshape(-1), fn.reshape(-1), s=2)

    m1 = min(np.min(fd), np.min(fn))
    m2 = max(np.max(fd), np.max(fn))
    if lim is not None:
        m1 = lim[0]
        m2 = lim[1]
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((fd-fn)**2))
    plt.text(np.min(fd) * 0.85 + np.max(fd) * 0.15, 
             np.min(fn) * 0.15 + np.max(fn) * 0.85,
             "RMSE: {:.3f} eV/A".format(rmse), fontsize=14)
    if symbol is None:
        plt.savefig('f.png')
    else:
        plt.savefig(f'{symbol}_f.png')
    return fig

def plot_v(vd, vn, lim=None, symbol=None):
    fig = plt.figure()
    plt.title("NEP virial vs DFT virial", fontsize=16)
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
    
    ax.set_xlabel('DFT virial (eV/atom)', fontsize=14)
    ax.set_ylabel('NEP virial (eV/atom)', fontsize=14)
    
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)    
    ax.tick_params(labelsize=16)

    idx = np.where(vn == -10**6)[0]  
    vd = np.delete(vd, idx) 
    vn = np.delete(vn, idx)

    plt.plot([np.min(vd), np.max(vd)], [np.min(vd), np.max(vd)],
            color='black',linewidth=3,linestyle='--',)
    plt.scatter(vd, vn, zorder=200)
    
    m1 = min(np.min(vd), np.min(vn))
    m2 = max(np.max(vd), np.max(vn))
    if lim is not None:
        m1 = lim[0]
        m2 = lim[1]

    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((vd-vn)**2))
    plt.text(np.min(vd) * 0.85 + np.max(vd) * 0.15, 
             np.min(vn) * 0.15 + np.max(vd) * 0.85,
             "RMSE: {:.3f} eV/atom".format(rmse), fontsize=14)
    if symbol is None:
        plt.savefig('v.png')
    else:
        plt.savefig(f'{symbol}_v.png')
    return fig


def plot_band_structure(atoms, formula, info):
    if 'band_dict' not in atoms.info:
        raise ValueError('No band structure data found.')
    os.makedirs('phonon', exist_ok=True)
    band_dict = atoms.info['band_dict']
    labels_path = band_dict['labels_path']
    frequencies = band_dict['frequencies']
    distances = band_dict['distances']
    fig = plt.figure()
    axs = ImageGrid(fig, 111, nrows_ncols=(1, len(labels_path)), axes_pad=0.2, label_mode="L")

    max_freq = max([np.max(fq) for fq in frequencies])
    max_dist = distances[-1][-1]
    xscale = max_freq / max_dist * 1.5
    distances_scaled = [d * xscale for d in distances]
    
    n = 0
    axs[0].set_ylabel("Frequency", fontsize=14)
    for i, path in enumerate(labels_path):
        axs[i].spines['bottom'].set_linewidth(1.5)
        axs[i].spines['left'].set_linewidth(1.5)
        axs[i].spines['right'].set_linewidth(1.5)
        axs[i].spines['top'].set_linewidth(1.5)
        axs[i].tick_params(labelsize=14)
        xticks = [distances_scaled[n][0]]
        for label in path[:-1]:
            xticks.append(distances_scaled[n][-1])
            axs[i].plot([distances_scaled[n][-1], distances_scaled[n][-1]], 
                        [0, max_freq],
                        linewidth=2,
                        linestyle=":",
                        c='grey')
            axs[i].plot(distances_scaled[n], 
                        frequencies[n], 
                        linewidth=2,
                        c='g')
            n += 1
        axs[i].set_xlim(xticks[0], xticks[-1])
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(path)
        axs[i].plot([xticks[0], xticks[-1]], 
                    [0, 0], 
                    linewidth=1,
                    c='black')
    
    fig.suptitle(f'{formula} {info} phonon dispersion', fontsize=16)
    fig_path = os.path.join('phonon' ,f'{formula}_{info}_phono.png')
    plt.savefig(fig_path)
    plt.close()
    return fig_path

def Prediction():
    e_1, e_2 = [], []
    v_1, v_2 = [], []
    f_x1, f_y1, f_z1, f_x2, f_y2, f_z2= [], [], [], [], [], []
    
    with open('energy_train.out', 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = list(map(float, line.strip().split()))
            if len(row) == 2:
                e_2.append(row[0])
                e_1.append(row[1])
    
    
    with open('virial_train.out', 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = list(map(float, line.strip().split()))
            if len(row) == 12:
                v_2.append(row[0])
                v_2.append(row[1])
                v_2.append(row[2])
                v_2.append(row[3])
                v_2.append(row[4])
                v_2.append(row[5])
                v_1.append(row[6])
                v_1.append(row[7])
                v_1.append(row[8])
                v_1.append(row[9])
                v_1.append(row[10])
                v_1.append(row[11])
    
    with open('force_train.out', 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = list(map(float, line.strip().split()))
            if len(row) == 6:
                f_x2.append(row[0])
                f_y2.append(row[1])
                f_z2.append(row[2])
                f_x1.append(row[3])
                f_y1.append(row[4])
                f_z1.append(row[5])
    
    e_1 = np.array(e_1)
    e_2 = np.array(e_2)
    e_rmse = np.sqrt(np.mean((e_1-e_2)**2)) 
    
    f_1 = f_x1 + f_y1 + f_z1
    f_2 = f_x2 + f_y2 + f_z2
    f_1 = np.array(f_1)
    f_2 = np.array(f_2)
    f_rmse = np.sqrt(np.mean((f_1-f_2)**2))  
    
    v_1 = np.array(v_1)
    v_2 = np.array(v_2)
    idx = np.where(v_2 == -10**6)[0]  
    v_1 = np.delete(v_1, idx) 
    v_2 = np.delete(v_2, idx)
    v_rmse = np.sqrt(np.mean((v_1-v_2)**2))
    
    plot_e(e_1, e_2)
    plot_f(f_1, f_2)
    plot_v(v_1, v_2)
    
    print(e_rmse)
    print(f_rmse)
    print(v_rmse)

def plot_thermo_out(filename, column=2, num=1):
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams.update({"font.size": 16, "text.usetex": False})
    fig, ax = plt.subplots()
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    lines = [line for line in lines if len(line) == 12]  
    lines = [[float(i) for i in line] for line in lines]
    data = np.array(lines)
    thermo = data[:, column] / num
    ax.plot(thermo, label=filename)
    ax.yaxis.set_major_formatter('{:.3f}'.format) 
    ax.set_xlabel('step(100)')  
    ax.set_ylabel('eV/atom')
    ax.legend()
    plt.tight_layout() 
    plt.savefig('thermo.png')

def plot_training_result(dirname = '', type = 'train'):
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams.update({"font.size": 10, "text.usetex": False})
    fig, axes = plt.subplots(2, 2)

    loss = np.loadtxt(dirname + "loss.out")
    energy = np.loadtxt(dirname + "energy_" + type + ".out")
    force = np.loadtxt(dirname + "force_" + type + ".out")
    virial_initial = np.loadtxt(dirname + "virial_" + type + ".out")
    virial= virial_initial[np.logical_not(np.any(virial_initial == -1e6, axis=1))]
    loss = loss[:, 2:7]

    axes[0, 0].loglog(loss)
    axes[0, 0].annotate("(a)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Loss functions")
    axes[0, 0].legend(["L1", "L2", "Energy", "Force", "Virial"], loc="lower left")

    axes[0, 1].plot(energy[:, 1], energy[:, 0], ".", markersize=10, color=[0, 0.45, 0.74])
    axes[0, 1].plot(axes[0, 1].get_xlim(), axes[0, 1].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
    axes[0, 1].annotate("(b)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
    axes[0, 1].set_xlabel("DFT energy (eV/atom)")
    axes[0, 1].set_ylabel("NEP energy (eV/atom)")
    rmse = np.sqrt(np.mean((energy[:, 0] - energy[:, 1])**2)) * 1000
    print(f"RMSE: {rmse:.3f} meV/atom")
    x_min = np.min(energy[:, 1]) * 1.1
    x_max = np.max(energy[:, 1]) * 1.1
    y_min = np.min(energy[:, 1]) * 1.1
    y_max = np.max(energy[:, 1]) * 1.1

    axes[0, 1].set_xlim(x_min, x_max)
    axes[0, 1].set_ylim(y_min, y_max)

    axes[1, 0].plot(force[:, 3:6], force[:, 0:3], ".", markersize=10, color=[0, 0.45, 0.74])
    axes[1, 0].plot(axes[1, 0].get_xlim(), axes[1, 0].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
    axes[1, 0].annotate("(c)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
    axes[1, 0].set_xlabel("DFT force (eV/Å)")
    axes[1, 0].set_ylabel("NEP force (eV/Å)")
    x_min = np.min(force[:, 3:6]) * 1.1
    x_max = np.max(force[:, 3:6]) * 1.1
    y_min = np.min(force[:, 3:6]) * 1.1
    y_max = np.max(force[:, 3:6]) * 1.1
    rmse = np.sqrt(np.mean((force[:, 0:3] - force[:, 3:6])**2)) * 1000
    print(f"RMSE: {rmse:.3f} meV/Å")

    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(y_min, y_max)

    cols = virial.shape[1]
    half = cols // 2
    axes[1, 1].plot(virial[:, :half], virial[:, half:], ".", markersize=10, color=[0, 0.45, 0.74])
    axes[1, 1].plot(axes[1, 1].get_xlim(), axes[1, 1].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
    axes[1, 1].annotate("(d)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
    axes[1, 1].set_xlabel("DFT virial (eV/atom)")
    axes[1, 1].set_ylabel("NEP virial (eV/atom)")
    x_min = np.min(virial[:, 0:6]) * 1.1
    x_max = np.max(virial[:, 0:6]) * 1.1
    y_min = np.min(virial[:, 0:6]) * 1.1
    y_max = np.max(virial[:, 0:6]) * 1.1
    rmse = np.sqrt(np.mean((virial[:, :half] - virial[:, half:])**2)) * 1000
    print(f"RMSE: {rmse:.3f} meV/atom")

    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(y_min, y_max)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.savefig("train_results.png")
    plt.show()

def plot_force_results(frames, calcs, labels = None, e_val = [None, None], f_val = [None, None]):
    plt.rcParams["figure.figsize"] = (12, 5)
    plt.rcParams.update({"font.size": 18, "text.usetex": False})
    fig, axes = plt.subplots(1, 2) 
    plt.subplots_adjust(wspace=0.3, bottom=0.2)
    cmap = plt.get_cmap("tab10")
  
    print(len(frames))  

    label_colors = {}
    if labels is None:
        labels = [str(i) for i in range(len(calcs))]
    for calc, label in zip(calcs, labels):
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
        color = cmap(labels.index(label))
        axes[0].plot(e_2, e_1, ".", markersize=10, label=label, color=color)
        axes[1].plot(f_2, f_1, ".", markersize=10, label=label, color=color)
        if label not in label_colors:
            label_colors[label] = color
        e_rmse = np.sqrt(np.mean((e_1-e_2)**2)) 
        f_rmse = np.sqrt(np.mean((f_1-f_2)**2))
        print(f'{label}_E_rmse: {e_rmse * 1000:.2f} meV/atom')
        print(f'{label}_F_rmse: {f_rmse * 1000:.2f} meV/Å')

    x_min, x_max = axes[0].get_xlim()
    y_min, y_max = axes[0].get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--')

    x_min, x_max = axes[1].get_xlim()
    y_min, y_max = axes[1].get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--')
    
    if e_val[0] is not None and e_val[1] is not None:
        axes[0].set_xlim(e_val)
        axes[0].set_ylim(e_val)
    if f_val[0] is not None and f_val[1] is not None:
        axes[1].set_xlim(f_val)
        axes[1].set_ylim(f_val)
    axes[0].set_xlabel("DFT energy (eV/atom)")
    axes[0].set_ylabel("NEP energy (eV/atom)")
    axes[1].set_xlabel("DFT force (eV/Å)")
    axes[1].set_ylabel("NEP force (eV/Å)")
    axes[0].text(0.05, 0.95, '(a)', transform=axes[0].transAxes, verticalalignment='top')
    axes[1].text(0.05, 0.95, '(b)', transform=axes[1].transAxes, verticalalignment='top')

    handles = [plt.Line2D([0], [0], color=color, marker='o', linestyle='') for label, color in label_colors.items()]
    plt.legend(handles, label_colors.keys(), loc = "upper right")
    plt.savefig("force_results.png")
    plt.show()
    plt.close()