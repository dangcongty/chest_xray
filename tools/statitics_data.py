from glob import glob

import matplotlib.pyplot as plt
import numpy as np

label_paths = glob('datasets/process/labels/*.txt')

class_dist = {i: 0 for i in range(14)}
for path in label_paths:
    with open(path, 'r') as f:
        data = f.readlines()

    for dta in data:
        c = int(dta.split(' ')[0])
        class_dist[c] += 1


classes = list(class_dist.keys())
counts = list(class_dist.values())

# Create the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(classes, counts, color='steelblue', edgecolor='black', alpha=0.7)

# Customize the chart
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(classes)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=9)

# Add total count
total = sum(counts)
plt.text(0.02, 0.98, f'Total: {total:,}', 
         transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('datasets/after_process.jpg')