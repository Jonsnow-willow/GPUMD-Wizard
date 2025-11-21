from ase import Atoms
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
        return key in atoms.info and atoms.info[key] is not None
    
    valid_keys = {key: is_valid_key(key) for key in ['energy', 'stress', 'forces', 'group', 'config_type', 'weight']}

    with open(filename, 'a') as f:
        Out_string = ""
        Out_string += str(int(len(atoms))) + "\n"
        Out_string += "pbc=\"" + " ".join(["T" if pbc_value else "F" for pbc_value in atoms.get_pbc()]) + "\" "
        Out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
        if valid_keys['energy']:
            Out_string += " energy=" + str(atoms.info['energy']) + " "
        if valid_keys['stress']:
            if len(atoms.info['stress']) == 6:
                virial = -atoms.info['stress'][[0, 5, 4, 5, 1, 3, 4, 3, 2]] * atoms.get_volume()
            else:
                virial = -atoms.info['stress'].reshape(-1) * atoms.get_volume()
            Out_string += "virial=\"" + " ".join(list(map(str, virial))) + "\" "
            Out_string += "stress=\"" + " ".join(list(map(str, atoms.info['stress'].reshape(-1)))) + "\" "

        Out_string += "Properties=species:S:1:pos:R:3:mass:R:1"
        if atoms.has('momenta'):
            velocites = atoms.get_velocities()
            Out_string += ":vel:R:3"
        if valid_keys['forces']:
            Out_string += ":force:R:3"
        if valid_keys['group']:
            group = atoms.info['group']
            num_atoms = len(atoms)
            if any(len(g) != num_atoms for g in group):
                raise ValueError(
                    f"Group data dimensions do not match number of atoms ({num_atoms})"
                )
            Out_string += f":group:I:{len(group)}"
        if valid_keys['config_type']:
            Out_string += f" config_type={atoms.info['config_type']}"
        if valid_keys['weight']:
            Out_string += " weight=" + f"{atoms.info['weight']:.2f}" 
        Out_string += "\n"
        for atom in atoms:
            Out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}'.format(atom.symbol, *atom.position, atom.mass)
            if atoms.has('momenta'):
                Out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*velocites[atom.index])
            if valid_keys['forces']:
                Out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.info['forces'][atom.index])
            if valid_keys['group']:
                for g in group:
                    Out_string += f" {int(g[atom.index])}"
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
        return [int(g) for g in group]
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
            if "stress=" in comment:
                stress = comment.split("stress=\"")[1].split("\"")[0].strip()
                stress = np.array([float(x) for x in stress.split()]).reshape(3, 3)
            elif "virial=" in comment:
                virials = comment.split("virial=\"")[1].split("\"")[0].strip()
                virials = np.array([float(x) for x in virials.split()]).reshape(3, 3)
                stress = - virials / np.linalg.det(cell)
            else:
                stress = None
            if "config_type=" in comment:
                config_type = comment.split("config_type=")[1].split()[0].strip()
            else:
                config_type = None
            if "weight=" in comment:
                weight = float(comment.split("weight=")[1].split()[0])
            else:
                weight = None
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
            if "force" not in comment:
                forces = None
            if "vel" not in comment:
                velocities = None
            if "group" in comment:
                group = [np.asarray(col, dtype=int) for col in zip(*group)] 
            else:
                group = None
            frames.append(Atoms(symbols=symbols, positions=positions, masses=masses, cell=cell, pbc=pbc, velocities=velocities, info={'energy': energy, 'stress': stress, 'forces': forces, 'group': group, 'config_type': config_type, 'weight': weight}))
    return frames