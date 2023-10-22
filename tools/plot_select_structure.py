"""
# -----------------------------------------------------------------------------
# The following code is sourced from
# pynep:
# https://github.com/bigd4/PyNEP
#
# Copyright (c) 2022 wang laosi, licensed under MIT License
# -----------------------------------------------------------------------------
"""
from pynep.calculate import NEP
from pynep.select import FarthestPointSample
from pynep.io import load_nep, dump_nep
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


atoms = load_nep('train.xyz', ftype="exyz")
calc = NEP('nep.txt')
des = np.array([np.mean(calc.get_property('descriptor', i), axis=0) for i in atoms])
sampler = FarthestPointSample(min_distance=0.008)
selected_i = sampler.select(des, [])
train_set = [atoms[i] for i in selected_i]
dump_nep('selected_train.xyz', train_set, ftype="exyz")

reducer = PCA(n_components=2)
reducer.fit(des)
proj = reducer.transform(des)
plt.scatter(proj[:,0], proj[:,1], label='all data')
selected_proj = reducer.transform(np.array([des[i] for i in selected_i]))
plt.scatter(selected_proj[:,0], selected_proj[:,1], label='selected data')
plt.legend()
plt.axis('off')
plt.savefig('select.png')

