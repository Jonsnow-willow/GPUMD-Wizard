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
import os
import random

def run_gpumd(dirname = './'):
    """
    Change the current working directory to the specified directory,
    run the 'gpumd' command, and then change the working directory back
    to the original directory.
    """
    original_directory = os.getcwd()
    os.chdir(dirname)
    os.system('gpumd')
    os.chdir(original_directory)

def mkdir_relax(atoms, run_in = ['potential ../nep.txt', 'velocity 300', 'time_step 1', 
                                 'ensemble npt_scr 300 300 200 0 500 2000', 'dump_thermo 1000', 
                                 'dump_restart 30000', 'run 30000']):
    if os.path.exists('relax'):
        raise FileExistsError('Directory "relax" already exists')
    os.makedirs('relax')
    original_directory = os.getcwd()
    os.chdir('relax')
    dump_xyz('model.xyz', atoms)
    write_run(run_in)
    os.chdir(original_directory)

def create_md(atoms, run_in = ['potential ../nep.txt', 'velocity 300', 'time_step 1', 
                                'ensemble npt_scr 300 300 200 0 500 2000', 
                                'dump_exyz 10000', 'run 1000000']):
    if not os.path.exists('md'):
        os.makedirs('md')
    else:
        if os.path.exists('md/model.xyz'):
            os.remove('md/model.xyz')
        if os.path.exists('md/run.in'):
            os.remove('md/run.in')
    original_directory = os.getcwd()
    os.chdir('md')
    dump_xyz('model.xyz', atoms)
    write_run(run_in)
    os.system('gpumd')
    os.chdir(original_directory)
    
def run_lammps(filename):
    """
    Run the LAMMPS simulation using the specified input file.
    """
    os.system('lmp_serial -in {}'.format(filename))

def write_run(parameters):
    """
    Write the input parameters to a gpumd file 'run.in'.
    """
    with open('run.in','w') as f:
        for i in parameters:
            f.write(i+'\n')

def dump_xyz(filename, atoms, comment = ''):
    """
    Append the atomic positions and other information to a file in XYZ format.
    """
    with open(filename, 'a') as f:
        Out_string = ""
        Out_string += str(int(len(atoms))) + "\n"
        Out_string += "pbc=\"" + " ".join(["T" if pbc_value else "F" for pbc_value in atoms.get_pbc()]) + "\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
        Out_string += "Properties=species:S:1:pos:R:3:mass:R:1"
        Out_string += comment + "\n"
        s = atoms.get_chemical_symbols()
        p = atoms.get_positions()
        m = atoms.get_masses()
        for j in range(int(len(atoms))):              
            Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}\n'.format(s[j], *p[j], m[j])
        f.write(Out_string)

def read_xyz(filename):
    """
    Read the atomic positions and other information from a file in XYZ format.
    """
    with open(filename) as f:
        lines = f.readlines()
        frames = []
        while lines:
            symbols = []
            positions = []
            natoms = int(lines.pop(0))
            comment = lines.pop(0)  # Comment line; ignored
            if "pbc=\"" in comment:
                pbc_str = comment.split("pbc=\"")[1].split("\"")[0].strip()
                pbc = [True if pbc_value == "T" else False for pbc_value in pbc_str.split()]
            else:
                pbc = [True, True, True]
            lattice_str = comment.split("Lattice=\"")[1].split("\" ")[0].strip()
            lattice = [list(map(float, row.split())) for row in lattice_str.split(" ")]
            cell = [lattice[0] + lattice[1] + lattice[2], lattice[3] + lattice[4] + lattice[5], lattice[6] + lattice[7] + lattice[8]]
            for _ in range(natoms):
                line = lines.pop(0)
                symbol, x, y, z = line.split()[:4]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
            if "energy=" in comment:
                energy = float(comment.split("energy=")[1].split()[0])
                frames.append(Atoms(symbols=symbols, positions=positions, cell = cell, pbc = pbc, info={"energy": energy}))
            else:
                frames.append(Atoms(symbols=symbols, positions=positions, cell = cell, pbc = pbc))
    return frames

def group_xyz(filename, atoms, min_xyz=[], max_xyz=[]):
    """
    Write the atomic positions and other information to a file in XYZ format,
    with an additional column indicating whether each atom is inside, outside,
    or on the boundary of a specified box.
    """
    with open(filename, 'w') as f:
        Out_string = ""
        Out_string += str(int(len(atoms))) + "\n"
        Out_string += "pbc=\"" + " ".join(["T" if pbc_value else "F" for pbc_value in atoms.get_pbc()]) + "\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
        Out_string += "Properties=species:S:1:pos:R:3:mass:R:1:group:I:1\n"
        s = atoms.get_chemical_symbols()
        p = atoms.get_positions()
        m = atoms.get_masses()
        for j in range(int(len(atoms))):
            if p[j][0] < min_xyz[0] or p[j][1] < min_xyz[1] or p[j][2] < min_xyz[2]:
                Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e} 0\n'.format(s[j], *p[j], m[j])
            elif  p[j][0] >= max_xyz[0] or p[j][1] >= max_xyz[1] or p[j][2] >= max_xyz[2]:
                Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e} 1\n'.format(s[j], *p[j], m[j])
            else: 
                Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e} 2\n'.format(s[j], *p[j], m[j])           
        f.write(Out_string)

def set_pka(input_restart, energy, angle, num, is_group = True):
    """
    This function reads in a file in a extxyz format, 
    modifies the velocities of the atoms in the file based on the input parameters, 
    and writes the modified data to a new file in a extxyz format.
    """
    with open(input_restart, 'r') as f:
        data = {}
        line = f.readline()
        data['header'] = line
        N = int(line.split(' ')[0])
        data['N'] = N
        data['box'] = f.readline()
        data['atom']={}
        data['velocity']=np.zeros([N,3], dtype=float)
        data['momentum']=np.zeros([N,3], dtype=float)
        data['group'] = {}
        for i in range(N):
            line = f.readline()
            atom = line.split(' ')
            data['atom'][i] = atom[0:5]
            data['velocity'][i, 0] = float(atom[5])
            data['velocity'][i, 1] = float(atom[6])
            data['velocity'][i, 2] = float(atom[7])
            data['momentum'][i, 0] = float(atom[5]) * float(atom[4])
            data['momentum'][i, 1] = float(atom[6]) * float(atom[4])
            data['momentum'][i, 2] = float(atom[7]) * float(atom[4])
            data['group'][i] = atom[8]
        
        kx = data['momentum'][:, 0].sum()
        ky = data['momentum'][:, 1].sum()
        kz = data['momentum'][:, 2].sum()
        print(kx, ky, kz)
        
        mass = float(data['atom'][num - 1][4])
        vx = pow(2 * energy / mass , 0.5) * angle[0] / pow(np.sum(angle ** 2), 0.5) / 10.18
        vy = pow(2 * energy / mass , 0.5) * angle[1] / pow(np.sum(angle ** 2), 0.5) / 10.18
        vz = pow(2 * energy / mass , 0.5) * angle[2] / pow(np.sum(angle ** 2), 0.5) / 10.18
        
        deltax = (mass * (vx - data['velocity'][num - 1, 0]))/ (N - 1)
        deltay = (mass * (vy - data['velocity'][num - 1, 1]))/ (N - 1)
        deltaz = (mass * (vz - data['velocity'][num - 1, 2]))/ (N - 1)
        
        for i in range(N):
            data['velocity'][i, 0] = data['velocity'][i, 0] - deltax / float(data['atom'][i][4])
            data['velocity'][i, 1] = data['velocity'][i, 1] - deltay / float(data['atom'][i][4])
            data['velocity'][i, 2] = data['velocity'][i, 2] - deltaz / float(data['atom'][i][4])

        data['velocity'][num - 1, 0] = vx
        data['velocity'][num - 1, 1] = vy
        data['velocity'][num - 1, 2] = vz
        
        kx = 0
        ky = 0
        kz = 0
        
        for i in range(N):
            kx = kx + data['velocity'][i, 0] * float(data['atom'][i][4])
            ky = ky + data['velocity'][i, 1] * float(data['atom'][i][4])
            kz = kz + data['velocity'][i, 2] * float(data['atom'][i][4])
        print(kx, ky, kz)
        
    with open('./model.xyz', 'w') as f:
        outstr =str(data['header']) + str(data['box'])
        if (is_group):
            for i in range(data['N']):
                outstr +=  '  '.join(data['atom'][i]) + ' ' + ' '.join(str(i) for i in data['velocity'][i]) + ' ' + ' '.join(data['group'][i]) + ' \n'
        else:
            for i in range(data['N']):
                outstr +=  '  '.join(data['atom'][i]) + ' ' + ' '.join(str(i) for i in data['velocity'][i]) + ' ' + ' '.join(data['group'][i])         
        f.write(outstr)

def plot_e(ed, er, lim=None, symbol=None):
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
    if lim is not None:
        m1 = lim[0]
        m2 = lim[1]
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((ed-er)**2))
    plt.text(np.min(ed) * 0.85 + np.max(ed) * 0.15, 
             np.min(er) * 0.15 + np.max(ed) * 0.85,
             "RMSE: {:.3f} eV/atom".format(rmse), fontsize=14)
    if symbol is None:
        plt.savefig('e.png')
    else:
        plt.savefig(f'{symbol}_e.png')
    return fig

def plot_f(fd, fr, lim=None, symbol=None):
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
    if symbol is None:
        plt.savefig('f.png')
    else:
        plt.savefig(f'{symbol}_f.png')
    return fig

def plot_v(vd, vr, lim=None, symbol=None):
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

    
    plt.plot([np.min(vd), np.max(vd)], [np.min(vd), np.max(vd)],
            color='black',linewidth=3,linestyle='--',)
    plt.scatter(vd, vr, zorder=200)
    
    m1 = min(np.min(vd), np.min(vr))
    m2 = max(np.max(vd), np.max(vr))
    if lim is not None:
        m1 = lim[0]
        m2 = lim[1]

    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((vd-vr)**2))
    plt.text(np.min(vd) * 0.85 + np.max(vd) * 0.15, 
             np.min(vr) * 0.15 + np.max(vd) * 0.85,
             "RMSE: {:.3f} eV/atom".format(rmse), fontsize=14)
    if symbol is None:
        plt.savefig('v.png')
    else:
        plt.savefig(f'{symbol}_v.png')
    return fig

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
    else:
        raise ValueError('Invalid relaxation method.')
    
    if model == 'qn':
        dyn = QuasiNewton(ucf)
    elif model == 'lbfgs':
        dyn = LBFGS(ucf)
    elif model == 'fire':
        dyn = FIRE(ucf)
    elif model == 'no_opt':
        return
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
    current_order = 1
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

def active_learning(frames, main_potential, potentials, error_min, error_max = 100, n = 1000000):
    Train_set = []
    for atoms in frames:
        diff = []
        atoms.calc = main_potential
        f_0 = np.concatenate(atoms.get_forces())
        for potential in potentials:
            atoms.calc = potential
            diff.append(np.concatenate(atoms.get_forces()) - f_0)
        diff_array = np.concatenate(diff)
        if np.any((diff_array > error_min) & (diff_array < error_max)):
            Train_set.append(atoms)
    if len(Train_set) > n:
        Train_set = random.sample(Train_set, n)
    return Train_set

def plot_thermo_out(filename, num = 1):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line.split() for line in lines]
    lines = [line for line in lines if len(line) == 12]  
    lines = [[float(i) for i in line] for line in lines]
    data = np.array(lines)
    thermo = data[:, 2] / num
    mpl.rcParams['font.size'] = 14
    plt.plot(thermo, 'r-')
    plt.gca().yaxis.set_major_formatter('{:.3f}'.format) 
    plt.xlabel('step(100)')  
    plt.ylabel('eV/atom') 
    plt.savefig('thermo.png')

def plot_training_result():
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams.update({"font.size": 10, "text.usetex": False})

    loss = np.loadtxt("loss.out")
    energy = np.loadtxt("energy_train.out")
    force = np.loadtxt("force_train.out")
    virial_initial = np.loadtxt("virial_train.out")
    virial= virial_initial[np.logical_not(np.any(virial_initial == -1e6, axis=1))]
    loss = loss[:, 2:7]

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].loglog(loss)
    axes[0, 0].annotate("(a)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Loss functions")
    axes[0, 0].legend(["L1", "L2", "Energy", "Force", "Virial"], loc="lower left")
    axes[0, 0].set_xlim([0, loss.shape[0]])

    axes[0, 1].plot(energy[:, 1], energy[:, 0], ".", markersize=10, color=[0, 0.45, 0.74])
    axes[0, 1].plot(axes[0, 1].get_xlim(), axes[0, 1].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
    axes[0, 1].annotate("(b)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
    axes[0, 1].set_xlabel("DFT energy (eV/atom)")
    axes[0, 1].set_ylabel("NEP energy (eV/atom)")
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

    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(y_min, y_max)

    axes[1, 1].plot(virial[:, 0:6], virial[:, 6:12], ".", markersize=10, color=[0, 0.45, 0.74])
    axes[1, 1].plot(axes[1, 1].get_xlim(), axes[1, 1].get_xlim(), linewidth=2, color=[0.85, 0.33, 0.1])
    axes[1, 1].annotate("(d)", xy=(0.0, 1.1), xycoords="axes fraction", va="top", ha="right")
    axes[1, 1].set_xlabel("DFT virial (eV/atom)")
    axes[1, 1].set_ylabel("NEP virial (eV/atom)")
    x_min = np.min(virial[:, 0:6]) * 1.1
    x_max = np.max(virial[:, 0:6]) * 1.1
    y_min = np.min(virial[:, 0:6]) * 1.1
    y_max = np.max(virial[:, 0:6]) * 1.1

    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(y_min, y_max)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.savefig("train_results.png")
    plt.show()