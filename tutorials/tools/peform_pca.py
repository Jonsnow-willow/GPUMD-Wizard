"""
# -----------------------------------------------------------------------------
# The following code is sourced from
# calorine:
# https://gitlab.com/materials-modeling/calorine
#
# Licensed under the Mozilla Public License 2.0 (MPL-2.0).
# See LICENSE in the project root or https://www.mozilla.org/en/MPL/2.0/
# -----------------------------------------------------------------------------
"""

from calorine.nep import get_descriptors
from sklearn.decomposition import PCA
from wizard.io import read_xyz
import matplotlib.pyplot as plt
import numpy as np

descriptors = []
n = 0
num = []

frames = read_xyz("train.xyz")
for atoms in frames:
    d = get_descriptors(atoms, "nep.txt")
    descriptors.append(d)
    n += len(atoms)
num.append(n)

frames = read_xyz("test.xyz")
for atoms in frames:
    d = get_descriptors(atoms, "nep.txt")
    descriptors.append(d)
    n += len(atoms)
num.append(n)

all_descriptors = np.concatenate(descriptors, axis=0)
print("Total number of atoms:", sum(num))
print("Number of descriptor components:", all_descriptors.shape[1])
print("Total number of atoms in train set:", num[0])
print("Total number of atoms in test set:", num[1])

pca = PCA(n_components=2)
pc = pca.fit_transform(all_descriptors)

p0 = pca.explained_variance_ratio_[0]
p1 = pca.explained_variance_ratio_[1]
print("Explained variance ratio of PC1: %.4f" % p0)
print("Explained variance ratio of PC2: %.4f" % p1)

fig, ax = plt.subplots(figsize=(4, 3))

ax.scatter(pc[:num[0], 0], pc[:num[0], 1], alpha=0.5, label='train set')
ax.scatter(pc[num[0]:, 0], pc[num[0]:, 1], alpha=0.5, label='test set')
ax.set_xlabel(f'PCA dimension 0')
ax.set_ylabel(f'PCA dimension 1')
ax.legend(frameon=False)

plt.tight_layout()
fig.savefig("pca.png")
plt.close(fig)