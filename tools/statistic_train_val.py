import os

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(list_file):
    with open(list_file, "r") as f:
        img_files = [x.strip() for x in f.readlines()]
    # label_files = [os.path.splitext(p)[0] + ".txt" for p in img_files]
    label_files = [p.replace('images', 'labels').replace('.png', '.txt') for p in img_files]
    
    all_classes, areas, ratios, boxes_per_img = [], [], [], []
    
    for lf in label_files:
        if not os.path.exists(lf): 
            continue
        with open(lf, "r") as f:
            lines = f.readlines()
        boxes_per_img.append(len(lines))
        
        for line in lines:
            cls, x, y, w, h = map(float, line.strip().split())
            if w < 0 or h < 0:
                continue
            all_classes.append(int(cls))
            areas.append(w * h)
            ratios.append(w / h if h > 0 else 0)
    
    return {
        "classes": all_classes,
        "areas": areas,
        "ratios": ratios,
        "boxes_per_img": boxes_per_img
    }

os.makedirs('datasets/train_val_statistic', exist_ok=True)

train_stats = load_dataset("datasets/process/train.txt")
val_stats   = load_dataset("datasets/process/val.txt")

# ---- 1. Class distribution (bar chart chung) ----
train_classes, train_counts = np.unique(train_stats["classes"], return_counts=True)
val_classes, val_counts     = np.unique(val_stats["classes"], return_counts=True)

all_classes = sorted(set(train_classes) | set(val_classes))
train_counts_dict = dict(zip(train_classes, train_counts))
val_counts_dict   = dict(zip(val_classes, val_counts))

train_counts = [train_counts_dict.get(c, 0) for c in all_classes]
val_counts   = [val_counts_dict.get(c, 0) for c in all_classes]

x = np.arange(len(all_classes))
width = 0.4

plt.figure(figsize=(10,5))
plt.bar(x - width/2, train_counts, width, label="train")
plt.bar(x + width/2, val_counts, width, label="val")
plt.xticks(x, all_classes)
plt.title("Class distribution")
plt.legend()
plt.savefig('datasets/train_val_statistic/class_distribution.jpg')

# ---- 2. Box area distribution (histogram chồng) ----
plt.figure(figsize=(10,5))
plt.hist(train_stats["areas"], bins=50, alpha=0.5, label="train", density=True)
plt.hist(val_stats["areas"], bins=50, alpha=0.5, label="val", density=True)
plt.title("Box area distribution")
plt.legend()
plt.title("Class distribution")
plt.savefig('datasets/train_val_statistic/area_distribution.jpg')


# ---- 3. Aspect ratio distribution (histogram chồng) ----
plt.figure(figsize=(10,5))
plt.hist(train_stats["ratios"], bins=50, alpha=0.5, label="train", density=True)
plt.hist(val_stats["ratios"], bins=50, alpha=0.5, label="val", density=True)
plt.title("Aspect ratio distribution")
plt.legend()
plt.savefig('datasets/train_val_statistic/ratio_distribution.jpg')


# ---- 4. Boxes per image (histogram chồng) ----
plt.figure(figsize=(10,5))
plt.hist(train_stats["boxes_per_img"], bins=30, alpha=0.5, label="train", density=True)
plt.hist(val_stats["boxes_per_img"], bins=30, alpha=0.5, label="val", density=True)
plt.title("Boxes per image distribution")
plt.legend()
plt.savefig('datasets/train_val_statistic/num_box_distribution.jpg')

