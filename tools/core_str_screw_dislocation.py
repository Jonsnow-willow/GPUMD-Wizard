import atomman as am
import numpy as np
from atomman.defect import DifferentialDisplacement
import matplotlib.pyplot as plt

def layer_color(atom_position, burgers):
    layer = np.round(atom_position[2] / np.linalg.norm(burgers) * 3) % 3 
    layer = int(layer)
    if layer == 0:
        return 'red'
    elif layer == 1:
        return 'green'
    elif layer == 2:
        return 'blue'

# Set the lattice constant and Burgers vector
alat = 3.185
burgers = np.array([alat/2, alat/2, alat/2])

# Load the relaxed structure
system_0 = am.load('poscar', 'POSCAR.0')
system_1 = am.load('poscar', 'POSCAR.1')

neighbors = system_0.neighborlist(cutoff=0.9 * alat)
# Calculate the differential displacement vectors
dd = DifferentialDisplacement(system_0, system_1, neighbors=neighbors, reference=0)
ddmax = np.linalg.norm(burgers)

params = {}
params['plotxaxis'] = 'x'
params['plotyaxis'] = 'y'
params['xlim'] = (0, 60)
params['ylim'] = (0, 20)
params['zlim'] = (-0.01, np.linalg.norm(burgers) + 0.01) # Should be one periodic width (plus a small cushion)
params['figsize'] = 10         # Only one value as the other is chosen to make plots "regular"
params['arrowwidth'] = 1/250    # Made bigger to make arrows easier to see
params['arrowscale'] = 1.7     # Typically chosen to make arrows of length ddmax touch the corresponding atom circles

#pos = system_0.atoms.pos
#colors = [layer_color(p, burgers) for p in pos]
dd.plot(burgers, ddmax, atomcmap='RdYlBu',**params)
plt.title('Parallel to Burgers')
plt.show()