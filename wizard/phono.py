import traceback
import numpy as np
import spglib
import ase
from ase import Atoms
from ase.cell import Cell
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.interface.phonopy_yaml import PhonopyYaml

def ase2phono(atoms):
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())

def phono2ase(cell):
    return Atoms(symbols=cell.get_chemical_symbols(),
                    scaled_positions=cell.get_scaled_positions(),
                    cell=cell.get_cell(),
                    pbc=True)

class PhonoCalc:
    def __init__(self, atoms, calc, dim='Auto', mesh=(10, 10, 10), t_step=10, t_max=0., t_min=0.):
        self.atoms = atoms
        self.calc = calc
        self.dim = dim
        self.mesh = mesh
        self.t_step = t_step
        self.t_max = t_max
        self.t_min = t_min
        self.results = {}
        try:
            print('Calculating force constants...')
            unitcell = ase2phono(self.atoms)
            if isinstance(self.dim, str):
                if self.dim == 'Auto':
                    lengths = self.atoms.cell.lengths()
                    supercell_matrix = np.round(10 / lengths).astype('int')
            else:
                supercell_matrix = self.dim

            self.phonon = Phonopy(
                unitcell=unitcell, 
                supercell_matrix=supercell_matrix,
                primitive_matrix='auto',
                )
            
            self.phonon.generate_displacements(distance=0.01)
            supercells = self.phonon.get_supercells_with_displacements()
            set_of_forces = []
            for cell in supercells:
                forces = self.calc.get_forces(phono2ase(cell))
                forces -= np.mean(forces, axis=0)
                set_of_forces.append(forces)
            set_of_forces = np.array(set_of_forces)
            self.phonon.produce_force_constants(forces=set_of_forces)
            self.results['force_constants'] = self.phonon.force_constants

            phpy_yaml = PhonopyYaml(settings=
                    {'force_sets': True,
                     'displacements': True,
                     'force_constants': True,
                     'born_effective_charge': True,
                     'dielectric_constant': True})
            phpy_yaml.set_phonon_info(self.phonon)

        except Exception as e:
            print(traceback.format_exc())
            print('Fail to collect force constants')

    def get_band_structure(self):
        lattice = self.atoms.get_cell().T
        positions = self.atoms.get_scaled_positions()
        numbers = self.atoms.get_atomic_numbers()
        atoms_cell = (lattice, positions, numbers)
        cell = Cell(spglib.find_primitive(atoms_cell)[0])
        special_points = cell.get_bravais_lattice().get_special_points()
        labels_path = ase.dft.kpoints.parse_path_string(cell.bandpath().path)
        labels, path = [], []
        for label_path in labels_path:
            p = []
            for l in label_path:
                labels.append(l)
                p.append(special_points[l].tolist())
            path.append(p)
        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
        self.phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
        bands_dict = self.phonon.get_band_structure_dict()
        bands_dict['labels_path'] = labels_path
        self.results['band_dict'] = bands_dict
        self.atoms.info.update(self.results)
 