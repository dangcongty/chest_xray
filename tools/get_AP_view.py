import json
import os
import shutil

with open('result_view.json', 'r') as f:
    data = json.load(f)

os.makedirs('datasets/notPA', exist_ok=True)
for img_path in data:
    view, score = data[img_path]
    if view == 'AP':
        shutil.copy(img_path, f'datasets/notPA/{os.path.basename(img_path)}')