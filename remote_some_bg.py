import numpy as np

with open('datasets/process/val_3k_bg.txt', 'r') as f:
    data = f.readlines()

_data = {
    'obj': [],
    'bg': []
}
for dta in data:
    dta_txt = dta.strip().replace('.png', '.txt').replace('images', 'labels')
    with open(dta_txt, 'r') as ff:
        anno = ff.readlines()

        if len(anno) == 0:
            _data['bg'].append(dta)
        else:
            _data['obj'].append(dta)


with open('datasets/process/val_3k_0k_bg.txt', 'w') as f:
    for path in _data['obj']:
        f.write(f'{path}')
        