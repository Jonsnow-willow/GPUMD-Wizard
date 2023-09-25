#from Pynep:https://github.com/bigd4/PyNEP/tree/master/pynep
from pynep.calculate import NEP
from wizard.io import read_xyz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train = read_xyz('train.xyz')
test = read_xyz('test.xyz')
calc = NEP("nep.txt")

des_train = np.array([np.mean(calc.get_property('descriptor', i), axis=0) for i in train])
reducer = PCA(n_components=2)
reducer.fit(des_train)
proj_train = reducer.transform(des_train)

des_test = np.array([np.mean(calc.get_property('descriptor', i), axis=0) for i in test])
reducer = PCA(n_components=2)
reducer.fit(des_test)
proj_test = reducer.transform(des_test)

plt.scatter(proj_train[:,0], proj_train[:,1], label='train data')
plt.scatter(proj_test[:,0], proj_test[:,1], label='test data')
plt.legend()
plt.axis('off')
plt.savefig('compare.png')