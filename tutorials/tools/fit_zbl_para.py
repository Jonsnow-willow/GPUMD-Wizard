from wizard.io import read_xyz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def quadratic(r, p1, p2, p3, p4, p5, p6):
    Z1 = 74
    Z2 = 74
    a = 0.46848 / (Z1 ** 0.23 + Z2 ** 0.23) 
    A = 14.399645 * Z1 * Z2
    x = r / a
    return A / r * (p1 * np.exp(-p2 * x) + p3 * np.exp(-p4 * x) + p5 * np.exp(-p6 * x)) 

dimer = read_xyz('dimer.xyz')
r = []
y = []
p0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.55]
bounds = ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [1, 1, 1, 1, 1, 1])
for atoms in dimer:
    r.append(np.linalg.norm(atoms[0].position - atoms[1].position))
    y.append(atoms.info['energy'])
r = np.array(r)
y = np.array(y)

popt, pcov = curve_fit(quadratic, r, y, p0=p0, bounds= bounds, maxfev=100000)
print(popt)
plt.scatter(r, y)
plt.plot(r, quadratic(r, *popt), color='red')
plt.show()