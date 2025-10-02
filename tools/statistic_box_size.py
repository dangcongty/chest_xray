from glob import glob

import matplotlib.pyplot as plt
import numpy as np

stats = {i: [] for i in range(14)}
for txt_path in glob('datasets/process/labels/*'):
    with open(txt_path, 'r') as f:
        data = f.readlines()
    
    for dt in data:
        dt = dt.strip().split()
        c = int(dt[0])
        x_center, y_center, w, h = np.array(dt[1:], dtype=float)
        w *= 1280
        h *= 1280
        stats[c].append([int(w), int(h)])

# Tính min width và min height cho từng class
min_ws, min_hs, labels = [], [], []
for c in sorted(stats.keys()):
    stat_c = np.array(stats[c]).reshape((-1, 2))
    stat_c = np.where(stat_c < 0, 0, stat_c)
    min_ws.append(stat_c[:,0].min())
    min_hs.append(stat_c[:,1].min())
    labels.append(f"Class {c}")

x = np.arange(len(labels))  # vị trí class
width = 0.35                # độ rộng của mỗi cột

fig, ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(x - width/2, min_ws, width, label='Min Width', color="orange")
rects2 = ax.bar(x + width/2, min_hs, width, label='Min Height', color="green")

ax.set_xlabel('Class')
ax.set_ylabel('Pixels')
ax.set_title('Minimum Width and Height per Class')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig('test.jpg')
