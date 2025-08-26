from ase import Atoms
from ase.neighborlist import NeighborList
from ase.optimize import QuasiNewton, FIRE, LBFGS
from ase.constraints import ExpCellFilter, FixedLine
from mpl_toolkits.axes_grid1 import ImageGrid    
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
import numpy as np

def write_run(parameters):
    """
    Write the input parameters to a gpumd file 'run.in'.
    """
    with open('run.in','w') as f:
        for i in parameters:
            f.write(i+'\n')

def dump_xyz(filename, atoms):
    def is_valid_key(key):
        return key in atoms.info and atoms.info[key] is not None and all(v is not None for v in atoms.info[key])
    
    valid_keys = {key: is_valid_key(key) for key in ['stress', 'velocities', 'forces', 'group']}

    with open(filename, 'a') as f:
        Out_string = ""
        Out_string += str(int(len(atoms))) + "\n"
        Out_string += "pbc=\"" + " ".join(["T" if pbc_value else "F" for pbc_value in atoms.get_pbc()]) + "\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
        if 'energy' in atoms.info and atoms.info['energy'] is not None:
            Out_string += " energy=" + str(atoms.info['energy']) + " "
        if valid_keys['stress']:
            if len(atoms.info['stress']) == 6:
                    virial = -atoms.info['stress'][[0, 5, 4, 5, 1, 3, 4, 3, 2]] * atoms.get_volume()
            else:
                virial = -atoms.info['stress'].reshape(-1) * atoms.get_volume()
            Out_string += "virial=\"" + " ".join(list(map(str, virial))) + "\" "
        Out_string += "Properties=species:S:1:pos:R:3:mass:R:1"
        if valid_keys['velocities']:
            Out_string += ":vel:R:3"
        if valid_keys['forces']:
            Out_string += ":force:R:3"
        if valid_keys['group']:
            Out_string += ":group:I:1"
        if 'config_type' in atoms.info and atoms.info['config_type'] is not None:
            Out_string += " config_type="+ atoms.info['config_type']
        if 'weight' in atoms.info and atoms.info['weight'] is not None:
            Out_string += " weight="+ str(atoms.info['weight'])
        Out_string += "\n"
        for atom in atoms:
            Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}'.format(atom.symbol, *atom.position, atom.mass)
            if valid_keys['velocities']:
                Out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.info['velocities'][atom.index])
            if valid_keys['forces']:
                Out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.info['forces'][atom.index])
            if valid_keys['group']:
                Out_string += ' {}'.format(atoms.info['group'][atom.index])
            Out_string += '\n'
        f.write(Out_string)

def parsed_properties(comment):
    properties_str = comment.split('properties=')[1].split()[0]
    properties = properties_str.split(':')
    parsed_properties = {}
    start = 0
    for i in range(0, len(properties), 3):
        property_name = properties[i]
        property_count = int(properties[i+2])
        parsed_properties[property_name] = slice(start, start + property_count)
        start += property_count
    return parsed_properties

def read_symbols(words_in_line, parsed_properties):
    symbol_slice = parsed_properties['species']
    symbol = words_in_line[symbol_slice]
    symbol = symbol[0].lower().capitalize()
    return symbol

def read_positions(words_in_line, parsed_properties):
    pos_slice = parsed_properties['pos']
    pos = words_in_line[pos_slice]
    return [float(pos[0]), float(pos[1]), float(pos[2])]

def read_mass(words_in_line, parsed_properties):
    if 'mass' in parsed_properties:
        mass_slice = parsed_properties['mass']
        mass = words_in_line[mass_slice]
        return float(mass[0])
    else:
        return None

def read_force(words_in_line, parsed_properties):
    force_key = 'forces' if 'forces' in parsed_properties else 'force'
    if force_key in parsed_properties:
        force_slice = parsed_properties[force_key]
        force = words_in_line[force_slice]
        return [float(force[0]), float(force[1]), float(force[2])]
    else:
        return None
    
def read_group(words_in_line, parsed_properties):
    if 'group' in parsed_properties:
        group_slice = parsed_properties['group']
        group = words_in_line[group_slice]
        return int(group[0])
    else:
        return None

def read_velocity(words_in_line, parsed_properties):
    if 'vel' in parsed_properties:
        vel_slice = parsed_properties['vel']
        vel = words_in_line[vel_slice]
        return [float(vel[0]), float(vel[1]), float(vel[2])]
    else:
        return None

def read_xyz(filename):
    """
    Read the atomic positions and other information from a file in XYZ format.
    """
    frames = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            symbols = []
            positions = []
            masses = []
            forces = []
            velocities = []
            group = []
            natoms = int(line.strip())
            comment = f.readline().lower().strip()
            if "pbc=\"" in comment:
                pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
                pbc = [True if pbc_value == "t" else False for pbc_value in pbc_str.split()]
            else:
                pbc = [True, True, True]
            lattice_str = comment.split("lattice=\"")[1].split("\"")[0].strip()
            lattice = [list(map(float, row.split())) for row in lattice_str.split(" ")]
            cell = [lattice[0] + lattice[1] + lattice[2], lattice[3] + lattice[4] + lattice[5], lattice[6] + lattice[7] + lattice[8]]
            if "energy=" in comment:
                energy = float(comment.split("energy=")[1].split()[0])
            else: 
                energy = None
            if "virial=" in comment:
                virials = comment.split("virial=\"")[1].split("\"")[0].strip()
                virials = np.array([float(x) for x in virials.split()]).reshape(3, 3)
                stress = - virials / np.linalg.det(cell)
            else:
                stress = None
            if "config_type=" in comment:
                config_type = comment.split("config_type=")[1].split()[0].strip()
            else:
                config_type = None
            parsed_properties_dict = parsed_properties(comment)
            for _ in range(natoms):
                line = f.readline()
                words_in_line = line.split()
                symbols.append(read_symbols(words_in_line, parsed_properties_dict))
                positions.append(read_positions(words_in_line, parsed_properties_dict))
                masses.append(read_mass(words_in_line, parsed_properties_dict))
                forces.append(read_force(words_in_line, parsed_properties_dict))
                velocities.append(read_velocity(words_in_line, parsed_properties_dict))
                group.append(read_group(words_in_line, parsed_properties_dict))
            frames.append(Atoms(symbols=symbols, positions=positions, masses=masses, cell=cell, pbc=pbc, info={'energy': energy, 'stress': stress, 'forces': forces, 'velocities': velocities, 'group': group, 'config_type': config_type}))
    return frames

def read_restart(filename):
    with open(filename, 'r') as f:
        line = f.readline()
        natoms = int(line.split(' ')[0])
        symbols = []
        positions = []
        masses = []
        velocities = []
        group = []
        comment = f.readline()  
        if "pbc=\"" in comment:
            pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
            pbc = [True if pbc_value == "T" else False for pbc_value in pbc_str.split()]
        else:
            pbc = [True, True, True]
        lattice_str = comment.split("Lattice=\"")[1].split("\"")[0].strip()
        lattice = [list(map(float, row.split())) for row in lattice_str.split(" ")]
        cell = [lattice[0] + lattice[1] + lattice[2], lattice[3] + lattice[4] + lattice[5], lattice[6] + lattice[7] + lattice[8]]
        if "group" in comment:
            for _ in range(natoms):
                line = f.readline()
                symbol, x, y, z, mass, vx, vy, vz, group_info= line.split()[:9]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                velocities.append([float(vx), float(vy), float(vz)])
                masses.append(mass)
                group.append(group_info)      
            atoms = Atoms(symbols=symbols, positions=positions, masses=masses, cell=cell, pbc=pbc, info={'velocities': velocities, 'group': group})
        else:
            for _ in range(natoms):
                line = f.readline()
                symbol, x, y, z, mass, vx, vy, vz= line.split()[:8]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                velocities.append([float(vx), float(vy), float(vz)])
                masses.append(mass)  
            atoms = Atoms(symbols=symbols, positions=positions, masses=masses, cell=cell, pbc=pbc, info={'velocities': velocities})
    return atoms

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
    fig_path = f'{formula}_{info}_phono.png'
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

def relax(atoms, fmax = 0.01, steps = 500, model = 'qn', method = 'hydro'):
    if method == 'fixed_line':
        constraint = [FixedLine(atom.index, direction=[0, 0, 1]) for atom in atoms]
        atoms.set_constraint(constraint)
        ucf = atoms
    elif method == 'hydro':
        ucf = ExpCellFilter(atoms, scalar_pressure=0.0, hydrostatic_strain=True) 
    elif method == 'ucf':
        ucf = atoms
    elif method == 'no_opt':
        return
    else:
        raise ValueError('Invalid relaxation method.')
    
    if model == 'qn':
        dyn = QuasiNewton(ucf)
    elif model == 'lbfgs':
        dyn = LBFGS(ucf)
    elif model == 'fire':
        dyn = FIRE(ucf)
    else:
        raise ValueError('Invalid optimization model.')
    
    dyn.run(fmax=fmax, steps=steps)

def get_nth_nearest_neighbor_index(atoms, index, nth):
    cutoffs = [5.0] * len(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    indices, offsets = neighbor_list.get_neighbors(index)
    distances = [atoms.get_distance(index, neighbor) for neighbor in indices]
    sorted_neighbors = sorted(zip(distances, indices), key=lambda x: x[0])
    current_order = 0
    nth_nearest_neighbor_index = None

    for i in range(len(sorted_neighbors)):
        if current_order == nth:
            nth_nearest_neighbor_index = sorted_neighbors[i][1]
            break
        if i < len(sorted_neighbors) - 1 and not np.isclose(sorted_neighbors[i][0], sorted_neighbors[i + 1][0]):
            current_order += 1
    return nth_nearest_neighbor_index

def symbol_to_string(symbols):
    element_counts = {}
    for element in symbols:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1
    result_string = ''
    for element, count in element_counts.items():
        result_string += f'{element}{count}'
    return result_string

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