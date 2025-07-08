import os
import numpy as np
import torch
from ase import io
from ase.neighborlist import neighbor_list

def find_neighbor(atoms, cutoff):
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    n_atoms = len(atoms)
    neighbors = [[] for _ in range(n_atoms)]
    distances = [[] for _ in range(n_atoms)]
    for idx in range(len(i)):
        neighbors[i[idx]].append(j[idx])
        distances[i[idx]].append(d[idx])
    return neighbors, distances