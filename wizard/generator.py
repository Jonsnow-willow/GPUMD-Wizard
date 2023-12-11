import numpy as np

class Generator:

    def __init__(self, frames):
        self.frames = frames

    def supercell(self, cell = (2, 2, 2)):
        for atoms in self.frames:
            atoms *= cell

    def deform(self, scale = np.arange(0.95, 1.06, 0.05)):
        frames = []
        for atoms in self.frames:
            for s in scale:
                atoms_copy = atoms.copy()
                atoms_copy.set_cell(atoms.get_cell() * s, scale_atoms=True)
                frames.append(atoms_copy)
        return frames
    
    def random_strain(self, ratio = 0.04):
        frames = []
        for atoms in self.frames:
            atoms_copy = atoms.copy()
            strain_matrix = np.eye(3) + 2 * ratio * (np.random.random((3,3)) - 0.5)
            new_cell = np.dot(atoms.get_cell(), strain_matrix)
            atoms_copy.set_cell(new_cell, scale_atoms=True)
            frames.append(atoms_copy)
        return frames
    
    def random_displacement(self, max_displacement = 0.4):
        frames = []
        for atoms in self.frames:
            atoms_copy = atoms.copy()
            atoms_copy.positions += np.random.uniform(-max_displacement, max_displacement, atoms_copy.positions.shape)
            frames.append(atoms_copy)
        return frames