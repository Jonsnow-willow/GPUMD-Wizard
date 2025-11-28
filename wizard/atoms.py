from ase import Atoms, Atom
from ase.build import bulk
from wizard.io import dump_xyz, write_run
import numpy as np
import random, os, re, shutil

class SymbolInfo:
    SUPPORTED_LATTICE_TYPES = {"bcc", "fcc", "hcp"}

    def __init__(self, formula, lattice_type, *lattice_constant):
        lattice_type = lattice_type.lower()
        if lattice_type not in self.SUPPORTED_LATTICE_TYPES:
            raise ValueError(
                f"Unsupported lattice type: {lattice_type}. Supported: {', '.join(sorted(self.SUPPORTED_LATTICE_TYPES))}"
            )
        self.formula = formula
        self.lattice_type = lattice_type
        self.lattice_constant = lattice_constant
        self.symbols = []
        self.compositions = []
        for symbol, composition in re.findall('([A-Z][a-z]*)(\d*)', formula):
            self.symbols.append(symbol)
            self.compositions.append(int(composition) if composition else 1)
    
    def create_bulk_atoms(self, supercell = (3, 3, 3)):
        symbol, crystalstructure, lc = self.symbols[0], self.lattice_type, self.lattice_constant
        if crystalstructure == 'hcp':
            atoms = bulk(symbol, crystalstructure, a = lc[0], c = lc[1]) * supercell
        else:
            atoms = bulk(symbol, crystalstructure, a = lc[0], cubic = True) * supercell
        if len(self.symbols) > 1:
            if len(atoms) < sum(self.compositions):
                raise ValueError('The number of atoms in the unit cell is less than the number of symbols.')
            element_ratio = np.array(self.compositions) / sum(self.compositions)
            element_counts = np.ceil(element_ratio * len(atoms)).astype(int)
            symbols = np.repeat(self.symbols, element_counts)
            np.random.shuffle(symbols)
            atoms.set_chemical_symbols(symbols[:len(atoms)])
        return atoms

    def __str__(self):
        return f"Formula: {self.formula}, Lattice Type: {self.lattice_type}, Lattice Constant: {self.lattice_constant}"

class Morph():
    def __init__(self, atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an instance of ase.Atoms")
        self.atoms = atoms
        
    def gpumd(self, dirname = 'relax', run_in = ['potential nep.txt', 'velocity 300', 'time_step 1', 
             'ensemble npt_scr 300 300 200 0 500 2000', 'dump_thermo 1000', 'dump_restart 30000', 
             'dump_exyz 10000','run 30000'], nep_path = 'nep.txt', gpumd_path = 'gpumd',
              electron_stopping_path = 'electron_stopping_fit.txt', run = True):
        atoms = self.atoms
        if os.path.exists(dirname):
            raise FileExistsError('Directory already exists')
        os.makedirs(dirname)
        if os.path.exists(nep_path):
            shutil.copy(nep_path, dirname)
        else:
            raise FileNotFoundError('nep.txt does not exist')
        if os.path.exists(electron_stopping_path):
            shutil.copy(electron_stopping_path, dirname)
        original_directory = os.getcwd()
        os.chdir(dirname)
        write_run(run_in)
        dump_xyz('model.xyz', atoms)
        if run:
            os.system(gpumd_path)
        os.chdir(original_directory)

    def set_pka(self, energy, direction, index = None, symbol = None):
        atoms = self.atoms
        direction = np.asarray(direction)
        if atoms.has('momenta') is None:
            raise ValueError('The velocities of atoms are not set.')
        velocites = atoms.get_velocities()
        
        if index is None:
            center = np.dot([0.5, 0.5, 0.5], atoms.get_cell())
            if symbol is None:
                index = np.argmin(np.sum((atoms.positions - center)**2, axis=1))
            else:
                element_indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
                element_positions = atoms.positions[element_indices]
                index = element_indices[np.argmin(np.sum((element_positions - center)**2, axis=1))]

        mass = atoms[index].mass
        vx = pow(2 * energy / mass , 0.5) * direction[0] / pow(np.sum(direction ** 2), 0.5) / 10.18051
        vy = pow(2 * energy / mass , 0.5) * direction[1] / pow(np.sum(direction ** 2), 0.5) / 10.18051
        vz = pow(2 * energy / mass , 0.5) * direction[2] / pow(np.sum(direction ** 2), 0.5) / 10.18051
        velocites[index] = [vx, vy, vz]
        
        atoms_masses = atoms.get_masses() 
        total_mass = np.sum(atoms_masses)
        momentum = np.sum(velocites * atoms_masses[:, np.newaxis], axis=0) 
        velocites -= momentum / total_mass
        atoms.set_velocities(velocites)

        print(f'Index: {index}')
        print(f'Symbol: {atoms[index].symbol}')
        print(f'Position: {atoms[index].position[0]:.2f}, {atoms[index].position[1]:.2f}, {atoms[index].position[2]:.2f}')
        print(f'Mass: {atoms[index].mass:.2f}')
        print(f'Velocity: {vx:.4f}, {vy:.4f}, {vz:.4f} (Angstrom/fs)')
       
    def velocity(self, vx, vy, vz, group = 0):
        atoms = self.atoms
        if atoms.has('momenta') is None:
            raise ValueError('The velocities of atoms are not set.')
        velocites = atoms.get_velocities()
        for index in range(len(atoms)):
            if int(atoms.info['group'][index]) == group:
                velocites[index] = [vx, vy, vz]
        
        atoms_masses = atoms.get_masses() 
        total_mass = np.sum(atoms_masses)
        momentum = np.sum(velocites * atoms_masses[:, np.newaxis], axis=0) 
        velocites -= momentum / total_mass
        atoms.set_velocities(velocites)

    def zero_momentum(self):
        atoms = self.atoms
        if atoms.has('momenta') is None:
            raise ValueError('The velocities of atoms are not set.')
        masses = atoms.get_masses()[:, np.newaxis]
        total_mass = np.sum(masses)
        momentum = atoms.get_momenta()
        total_momentum = np.sum(momentum, axis=0)
        momentum -= masses * (total_momentum / total_mass)
        atoms.set_momenta(momentum)
    
    def shuffle_symbols(self):
        atoms = self.atoms
        s = atoms.get_chemical_symbols()
        random.shuffle(s)
        atoms.set_chemical_symbols(s)

    def coord_element_set(self, coord, symbol):
        atoms = self.atoms
        for atom in atoms:
            if np.allclose(atom.position, coord):
                atom.symbol = symbol
                break

    def random_center(self, index = None):
        atoms = self.atoms
        if index is None:
            index = np.random.randint(0, len(atoms))
        center = atoms.cell.diagonal() / 2
        diff = center - atoms[index].position
        for atom in atoms:
            atom.position += diff

        for atom in atoms:
            atom.position %= atoms.cell.diagonal()

    def scale_lattice(self, scale):
        atoms = self.atoms
        origin_cell = atoms.cell.copy()
        atoms.set_cell(scale * origin_cell, scale_atoms=True)

    def create_self_interstitial_atom(self, vector, symbol = None, index = 0):
        atoms = self.atoms
        if symbol is not None:
            atom = Atom(symbol, atoms[index].position - vector)
        else:
            atom = Atom(atoms[index].symbol, atoms[index].position - vector)
        atoms[index].position += vector
        atoms.append(atom)

    def create_random_interstitial(self, symbols, num=1):
        atoms_to_insert = []
        for _ in range(num):
            symbol = random.choice(symbols)
            atom = Atom(symbol=symbol)
            atoms_to_insert.append(atom)
        self.insert_atoms(atoms_to_insert)

    def create_vacancy(self, index = 0):
        del self.atoms[index]

    def create_vacancies(self, num_vacancies):
        atoms = self.atoms
        if num_vacancies > len(atoms):
            raise ValueError("num_vacancies should be less than or equal to the total number of atoms.")
        indices_to_remove = np.random.choice(len(atoms), num_vacancies, replace=False)
        removed_atoms = atoms[indices_to_remove]
        indices_to_remove = sorted(indices_to_remove, reverse=True)
        for index in indices_to_remove:
            del self.atoms[index]
        return removed_atoms
    
    def insert_atoms(self, atoms_to_insert, distance=1.2):
        indices_to_insert = np.random.choice(len(self.atoms), len(atoms_to_insert), replace=False)
        target_atoms = self.atoms[indices_to_insert]
        for atom_to_insert, target_atom in zip(atoms_to_insert, target_atoms):
            unit_vector = np.random.randn(3)  
            unit_vector /= np.linalg.norm(unit_vector)  
            displacement_vector = distance * unit_vector  
            new_position = target_atom.position + displacement_vector 
            atom_to_insert.position = new_position
            self.atoms.append(atom_to_insert)
            
    def create_fks(self, num_vacancies):
        removed_atoms = self.create_vacancies(num_vacancies)
        self.insert_atoms(removed_atoms)