from ase import Atoms, Atom
from ase.build import bulk
from ase.optimize import QuasiNewton, FIRE, LBFGS
from ase.constraints import ExpCellFilter, FixedLine
from wizard.io import get_nth_nearest_neighbor_index, dump_xyz, write_run
import numpy as np
import random
import os
import re
import shutil

class SymbolInfo:
    def __init__(self, formula, structure, *lattice_constant):
        self.formula = formula
        self.structure = structure
        self.lattice_constant = lattice_constant
        self.symbols = []
        self.compositions = []
        for symbol, composition in re.findall('([A-Z][a-z]*)(\d*)', formula):
            self.symbols.append(symbol)
            self.compositions.append(int(composition) if composition else 1)
    
    def create_bulk_atoms(self, supercell = (3, 3, 3)):
        symbol, structure, lc = self.symbols[0], self.structure, self.lattice_constant
        if structure == 'hcp':
            atoms = bulk(symbol, structure, a = lc[0], c = lc[1]) * supercell
        else:
            atoms = bulk(symbol, structure, a = lc[0], cubic=True) * supercell
        if len(self.symbols) > 1:
            if len(atoms) < sum(self.compositions):
                raise ValueError('The number of atoms in the unit cell is less than the number of symbols.')
            element_ratio = np.array(self.compositions) / sum(self.compositions)
            element_counts = np.ceil(element_ratio * len(atoms)).astype(int)
            symbols = np.repeat(self.symbols, element_counts)
            np.random.shuffle(symbols)
            atoms.set_chemical_symbols(symbols[:len(atoms)])
        return atoms
    
class Morph():
    def __init__(self, atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an instance of ase.Atoms")
        self.atoms = atoms
        
    def relax(self, fmax = 0.01, steps = 500, model = 'qn', method = 'hydro'):
        atoms = self.atoms
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

    def gpumd(self, dirname = 'relax', run_in = ['potential nep.txt', 'velocity 300', 'time_step 1', 
             'ensemble npt_scr 300 300 200 0 500 2000', 'dump_thermo 1000', 'dump_restart 30000', 
             'dump_exyz 10000','run 30000'], nep_path = 'nep.txt', write_in = False):
        atoms = self.atoms
        if os.path.exists(dirname):
            raise FileExistsError('Directory already exists')
        os.makedirs(dirname)
        if os.path.exists(nep_path):
            shutil.copy(nep_path, dirname)
        else:
            raise FileNotFoundError('nep.txt does not exist')
        original_directory = os.getcwd()
        os.chdir(dirname)
        write_run(run_in)
        dump_xyz('model.xyz', atoms)
        if write_in:
           pass
        else:
            os.system('gpumd')
        os.chdir(original_directory)

    def set_pka(self, energy, direction, index = None, symbol = None):
        atoms = self.atoms
        if atoms.info['velocities'] is None:
            raise ValueError('The velocities of atoms are not set.')
        
        if index is None:
            center = np.diag(atoms.get_cell()) / 2.0
            if symbol is None:
                index = np.argmin(np.sum((atoms.positions - center)**2, axis=1))
            else:
                element_indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
                element_positions = atoms.positions[element_indices]
                index = element_indices[np.argmin(np.sum((element_positions - center)**2, axis=1))]

        mass = atoms[index].mass
        vx = pow(2 * energy / mass , 0.5) * direction[0] / pow(np.sum(direction ** 2), 0.5) / 10.18
        vy = pow(2 * energy / mass , 0.5) * direction[1] / pow(np.sum(direction ** 2), 0.5) / 10.18
        vz = pow(2 * energy / mass , 0.5) * direction[2] / pow(np.sum(direction ** 2), 0.5) / 10.18
        delta_momentum = (np.array(atoms.info['velocities'][index]) - np.array([vx, vy, vz])) * mass / (len(atoms) - 1)
        
        atoms_masses = np.array(atoms.get_masses())
        atoms.info['velocities'] += delta_momentum / atoms_masses[:, np.newaxis]
        atoms.info['velocities'][index] = [vx, vy, vz]
        
    def velocity(self, vx, vy, vz, group = 0):
        atoms = self.atoms
        if atoms.info['velocities'] is None:
            raise ValueError('The velocities of atoms are not set.')
        for index in range(len(atoms)):
            if int(atoms.info['group'][index]) == group:
                atoms.info['velocities'][index] = [vx, vy, vz]
        
        atoms_masses = np.array(atoms.get_masses())
        momentum = np.sum(atoms.info['velocities'] * atoms_masses[:, np.newaxis], axis=0) / len(atoms)
        atoms.info['velocities'] -= momentum / atoms_masses[:, np.newaxis]
    
    def shuffle_symbols(self):
        atoms = self.atoms
        s = atoms.get_chemical_symbols()
        random.shuffle(s)
        atoms.set_chemical_symbols(s)

    def prop_element_set(self, symbols = []):
        atoms = self.atoms
        n_atoms = len(atoms)
        n_sym = len(symbols)
        n_repeat, n_extra = divmod(n_atoms, n_sym)
        sym = np.array(symbols * n_repeat + symbols[:n_extra], dtype=str)
        np.random.shuffle(sym)
        atoms.set_chemical_symbols(sym.tolist())

    def element_random_replace(self, symbol1, symbol2, num):
        atoms = self.atoms
        symbols = atoms.get_chemical_symbols()
        indices = np.where(np.array(symbols) == symbol1)[0]
        if len(indices) < num:
            raise ValueError('The number of atoms to be replaced is greater than the number of atoms of the first element.')
        np.random.shuffle(indices)
        indices = indices[:num]
        for index in indices:
            atoms[index].symbol = symbol2

    def coord_element_set(self, coord, symbol):
        atoms = self.atoms
        for atom in atoms:
            if np.allclose(atom.position, coord):
                atom.symbol = symbol
                break

    def scale_lattice(self, scale):
        atoms = self.atoms
        origin_cell = atoms.cell.copy()
        atoms.set_cell(scale * origin_cell, scale_atoms=True)

    def coord_vac_set(self, coord):
        atoms = self.atoms.copy()
        for atom in atoms:
            if np.allclose(atom.position, coord):
                index = atom.index
                break
        del atoms[index]

    def create_interstitial(self, new_atom):
        atoms = self.atoms
        atoms.append(new_atom)

    def create_self_interstitial_atom(self, vector, symbol = None, index = 0):
        atoms = self.atoms
        if symbol is not None:
            atom = Atom(symbol, atoms[index].position - vector)
        else:
            atom = Atom(atoms[index].symbol, atoms[index].position - vector)
        atoms[index].position += vector
        atoms.append(atom)

    def create_di_self_interstitial_atoms(self, vector1, vector2, symbol1 = None, symbol2 = None, index = 0, nth = 1):
        atoms = self.atoms
        index_neibor = get_nth_nearest_neighbor_index(self.atoms, index, nth)
        if symbol1 is not None:
            atom1 = Atom(symbol1, atoms[index].position - vector1)
        else:
            atom1 = Atom(atoms[index].symbol, atoms[index].position - vector1)
        if symbol2 is not None:
            atom2 = Atom(symbol2, atoms[index_neibor].position - vector2)
        else:
            atom2 = Atom(atoms[index_neibor].symbol, atoms[index_neibor].position - vector2)
        atoms[index].position += vector1
        atoms[index_neibor].position += vector2
        atoms.append(atom1)
        atoms.append(atom2)

    def create_vacancy(self, index = 0):
        del self.atoms[index]

    def create_divacancies(self, index1 = 0, nth = 1):
        index2 = get_nth_nearest_neighbor_index(self.atoms, index1, nth)
        if index2 < index1:
            index1, index2 = index2, index1
        del self.atoms[index2]
        del self.atoms[index1]

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
            unit_vector = np.random.randn(3)  # Generate a random vector
            unit_vector /= np.linalg.norm(unit_vector)  # Normalize the vector to get a unit vector
            displacement_vector = distance * unit_vector  # Scale the unit vector to the desired distance
            new_position = target_atom.position + displacement_vector  # Add the displacement vector to the target atom's position
            atom_to_insert.position = new_position
            self.atoms.append(atom_to_insert)
            
    def create_fks(self, num_vacancies):
        removed_atoms = self.create_vacancies(num_vacancies)
        self.insert_atoms(removed_atoms)
    
    def get_atoms(self):
        return self.atoms.copy()
    
    def get_potential_energy(self):
        return self.atoms.get_potential_energy()