import json

with open('datasets/process/train_3k_bg.txt', 'r') as f:
    train_raw_paths = f.readlines()
with open('datasets/process/val_3k_bg.txt', 'r') as f:
    val_raw_paths = f.readlines()

with open('datasets/outliers.txt', 'r') as f:
    outliers = f.readlines()

with open('datasets/process/train_3k_bg_outlier.txt', 'w') as f:
    for path in train_raw_paths:
        _path = path.replace('images', 'labels').replace('.png', '.txt')
        if _path in outliers:
            print(_path)
            continue
        else:
            f.write(path)


with open('datasets/process/val_3k_bg_outlier.txt', 'w') as f:
    for path in val_raw_paths:
        _path = path.replace('images', 'labels').replace('.png', '.txt')
        if _path in outliers:
            print(_path)
            continue
        else:
            f.write(path)