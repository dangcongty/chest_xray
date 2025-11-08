# Visualization
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

for path in glob('datasets/heatmap/*'):
    heatmap = np.load(path)
    if heatmap.sum() < 1:
        continue
    print(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    image_shape = (640, 640)
    # Plot 1: Heatmap with bounding boxes
    axes[0].imshow(heatmap, cmap='hot', origin='upper')
    axes[0].set_title('Gaussian Heatmap with Bounding Boxes')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # Plot 2: 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D

    ax = fig.add_subplot(122, projection='3d')
    x = np.arange(0, image_shape[1], 5)
    y = np.arange(0, image_shape[0], 5)
    X, Y = np.meshgrid(x, y)
    Z = heatmap[::5, ::5]
    surf = ax.plot_surface(X, Y, Z, cmap='hot', alpha=0.8)
    ax.set_title('3D Heatmap Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    fig.colorbar(surf, ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.show()