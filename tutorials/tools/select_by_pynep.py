"""
# -----------------------------------------------------------------------------
# The following code is sourced from
# pynep:
# https://github.com/bigd4/PyNEP
#
# Copyright (c) 2022 wang laosi, licensed under MIT License
# -----------------------------------------------------------------------------
"""

from wizard.io import read_xyz, dump_xyz
from calorine.nep import get_descriptors
from scipy.spatial.distance import cdist
import numpy as np

class FarthestPointSample:
    def __init__(self, min_distance=0.1, metric='euclidean', metric_para={}):
        self.min_distance = min_distance
        self.metric = metric
        self.metric_para = {}

    def select(self, new_data, now_data=[], min_distance=None, min_select=1, max_select=None):
        min_distance = min_distance or self.min_distance
        max_select = max_select or len(new_data)
        to_add = []
        if len(new_data) == 0:
            return to_add
        if len(now_data) == 0:
            to_add.append(0)
            now_data.append(new_data[0])
        distances = np.min(cdist(new_data, now_data, metric=self.metric, **self.metric_para), axis=1)

        while np.max(distances) > min_distance or len(to_add) < min_select:
            i = np.argmax(distances)
            to_add.append(i)
            if len(to_add) >= max_select:
                break
            distances = np.minimum(distances, cdist([new_data[i]], new_data, metric=self.metric)[0])
        return to_add

def main():
    frames = read_xyz("test.xyz")
    descriptors = np.array([np.mean(get_descriptors(atoms, "nep.txt"), axis=0) for atoms in frames])
    sampler = FarthestPointSample(min_distance=0.05)
    selected = sampler.select(descriptors, [])
    for atoms in selected:
        dump_xyz("train.xyz", atoms)

if __name__ == "__main__":
    main()