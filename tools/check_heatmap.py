from glob import glob

import numpy as np

_max = 0
_min = np.inf
for path in glob('datasets/heatmap_v2/*'):
    hm = np.load(path)
    if hm.max() > _max:
        _max = hm.max()

    if hm.min() < _min:
        _min = hm.min()


print(f"Max: {_max} | Min: {_min}")