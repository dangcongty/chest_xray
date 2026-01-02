

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

model_path = 'runs/heatmap/48f693ea-cb15-487d-a573-4442d095e745_contrastive/weights/best.pt'
model = YOLO(model_path)

with open('datasets/process/val_1k_bg.txt', 'r') as f:
    paths = f.readlines()

embs = []
gts = []
for path in tqdm(paths):
    # if 'd8c4c835692b1951a853152460c04dc0' not in path:
    #     continue
    path = path.strip()
    with open(path.replace('images', 'labels').replace('png', 'txt'), 'r') as f:
        lb = f.readlines()

    res = model.predict(source=path)
    
    if (len(lb) != len(res[0])):
        print()