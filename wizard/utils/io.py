from ase import Atoms
import numpy as np

def write_run(parameters: list[str]):
    """
    Write the input parameters to a gpumd file 'run.in'.
    """
    with open('run.in','w') as f:
        for i in parameters:
            f.write(i+'\n')


def dump_xyz(filename: str, atoms: Atoms):
    """
    Write the atomic positions and other information to a file in XYZ format.
    """
    def is_valid_key(key: str) -> bool:
        return key in atoms.info and atoms.info[key] is not None
    
    valid_keys = {key: is_valid_key(key) for key in ['energy', 'stress', 'forces', 'group', 'config_type', 'weight', 'mag']}
    with open(filename, 'a') as f:
        out_string = ""
        out_string += str(int(len(atoms))) + "\n"
        out_string += "pbc=\"" + " ".join(["T" if pbc_value else "F" for pbc_value in atoms.get_pbc()]) + "\" "
        out_string += "Lattice=\"" + " ".join(list(map(str, atoms.get_cell().reshape(-1)))) + "\" "
        if valid_keys['energy']:
            out_string += " energy=" + str(atoms.info['energy']) + " "
        if valid_keys['stress']:
            if len(atoms.info['stress']) == 6:
                virial = -atoms.info['stress'][[0, 5, 4, 5, 1, 3, 4, 3, 2]] * atoms.get_volume()
                stress = atoms.info['stress'][[0, 5, 4, 5, 1, 3, 4, 3, 2]]
            else:
                virial = -atoms.info['stress'].reshape(-1) * atoms.get_volume()
                stress = atoms.info['stress'].reshape(-1)
            out_string += "virial=\"" + " ".join(list(map(str, virial))) + "\" "
            out_string += "stress=\"" + " ".join(list(map(str, stress))) + "\" "
        out_string += "Properties=species:S:1:pos:R:3:mass:R:1"
        if atoms.has('momenta'):
            velocites = atoms.get_velocities()
            out_string += ":vel:R:3"
        if valid_keys['forces']:
            out_string += ":force:R:3"
        if valid_keys['mag']:
            mag = atoms.info['mag']
            num_atoms = len(atoms)
            if len(mag) != num_atoms:
                raise ValueError(
                    f"Mag data dimensions do not match number of atoms ({num_atoms})"
            )
            mag_count = len(np.asarray(mag[0]).reshape(-1))
            if mag_count not in [1, 3]:
                raise ValueError("Mag data should have 1 or 3 components")
            out_string += f":mag:R:{mag_count}"
        if valid_keys['group']:
            group = atoms.info['group']
            num_atoms = len(atoms)
            if any(len(g) != num_atoms for g in group):
                raise ValueError(
                    f"Group data dimensions do not match number of atoms ({num_atoms})"
                )
            out_string += f":group:I:{len(group)}"
        if valid_keys['config_type']:
            out_string += f" config_type={atoms.info['config_type']}"
        if valid_keys['weight']:
            out_string += " weight=" + f"{atoms.info['weight']:.2f}" 
        out_string += "\n"
        for atom in atoms:
            out_string += '{:2} {:>15.8e} {:>15.8e} {:>15.8e} {:>15.8e}'.format(atom.symbol, *atom.position, atom.mass)
            if atoms.has('momenta'):
                out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*velocites[atom.index])
            if valid_keys['forces']:
                out_string += ' {:>15.8e} {:>15.8e} {:>15.8e}'.format(*atoms.info['forces'][atom.index])
            if valid_keys['mag']:
                for m in np.asarray(mag[atom.index]).reshape(-1):
                    out_string += ' {:>15.8e}'.format(m)
            if valid_keys['group']:
                for g in group:
                    out_string += f" {int(g[atom.index])}"
            out_string += '\n'
        f.write(out_string)

def _parsed_properties(comment: str) -> dict[str, slice]:
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

def _read_symbols(words_in_line: list[str], parsed_properties: dict[str, slice]) -> str:
    symbol_slice = parsed_properties['species']
    symbol = words_in_line[symbol_slice]
    symbol = symbol[0].lower().capitalize()
    return symbol

def _read_positions(words_in_line: list[str], parsed_properties: dict[str, slice]) -> tuple[float, float, float]:
    pos_slice = parsed_properties['pos']
    pos = words_in_line[pos_slice]
    return tuple(float(p) for p in pos)

def _read_mass(words_in_line: list[str], parsed_properties: dict[str, slice]) -> float | None:
    if 'mass' in parsed_properties:
        mass_slice = parsed_properties['mass']
        mass = words_in_line[mass_slice]
        return float(mass[0])
    else:
        return None

def _read_force(words_in_line: list[str], parsed_properties: dict[str, slice]) -> tuple[float, float, float] | None:
    force_key = 'forces' if 'forces' in parsed_properties else 'force'
    if force_key in parsed_properties:
        force_slice = parsed_properties[force_key]
        force = words_in_line[force_slice]
        return tuple(float(f) for f in force)
    else:
        return None
    
def _read_velocity(words_in_line: list[str], parsed_properties: dict[str, slice]) -> tuple[float, float, float] | None:
    if 'vel' in parsed_properties:
        vel_slice = parsed_properties['vel']
        vel = words_in_line[vel_slice]
        return tuple(float(v) for v in vel)
    else:
        return None
    
def _read_group(words_in_line: list[str], parsed_properties: dict[str, slice]) -> list[int] | None:
    if 'group' in parsed_properties:
        group_slice = parsed_properties['group']
        group = words_in_line[group_slice]
        return [int(g) for g in group]
    else:
        return None

def _read_mag(words_in_line: list[str], parsed_properties: dict[str, slice]) -> float | tuple[float, ...] | None:
    if 'mag' in parsed_properties:
        mag = words_in_line[parsed_properties['mag']]
        if len(mag) not in [1, 3]:
            raise ValueError("Mag data should have 1 or 3 components")
        mag = tuple(float(m) for m in mag)
        return mag[0] if len(mag) == 1 else mag
    else:
        return None

def read_xyz(filename: str) -> list[Atoms]:
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
            mag = []
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
            parsed_properties_dict = _parsed_properties(comment)
            for _ in range(natoms):
                line = f.readline()
                words_in_line = line.split()
                symbols.append(_read_symbols(words_in_line, parsed_properties_dict))
                positions.append(_read_positions(words_in_line, parsed_properties_dict))
                masses.append(_read_mass(words_in_line, parsed_properties_dict))
                forces.append(_read_force(words_in_line, parsed_properties_dict))
                velocities.append(_read_velocity(words_in_line, parsed_properties_dict))
                group.append(_read_group(words_in_line, parsed_properties_dict))
                mag.append(_read_mag(words_in_line, parsed_properties_dict))
            if "force" not in comment:
                forces = None
            if "vel" not in comment:
                velocities = None
            if "group" in comment:
                group = [np.asarray(col, dtype=int) for col in zip(*group)] 
            else:
                group = None
            if "mag" in comment:
                mag = np.asarray(mag)
            else:
                mag = None
            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                masses=masses,
                cell=cell,
                pbc=pbc,
                velocities=velocities,
                info={'energy': energy, 'stress': stress, 'forces': forces, 'group': group, 'config_type': config_type, 'weight': weight, 'mag': mag})
            frames.append(atoms)
    return frames
