import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

from get_embedding import extract_embeddings, load_class_mapping, load_model, PadToSquare
from torchvision import transforms
from PIL import Image

def assign_cluster(embeddings, labels, embedding_new):
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        centroid = embeddings[labels == label].mean(axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids)

    # Tính khoảng cách Euclidean
    distances = np.linalg.norm(centroids - embedding_new, axis=1)
    cluster_idx = np.argsort(distances)
    return unique_labels[cluster_idx], distances


# model, device = load_model('ckpt/best_embedding.pth', embedding_dim = 512)
embeddings_train = np.load('ckpt/dataX.npy')
labels_train = np.load('ckpt/dataY.npy')

embeddings_test = np.load('ckpt/dataX_test.npy')
labels_test = np.load('ckpt/dataY_test.npy')

class_mapping = load_class_mapping()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Circle
import os

# Tạo thư mục lưu hình
os.makedirs("tsne_per_class", exist_ok=True)

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

# Giả sử có:
# embeddings_train, labels_train
# embeddings_test,  labels_test
# class_mapping['idx_to_class']

# Gộp cả hai bộ vào cùng không gian t-SNE
X = np.concatenate([embeddings_train, embeddings_test], axis=0)
y = np.concatenate([labels_train, labels_test], axis=0)
domain = np.array([0]*len(embeddings_train) + [1]*len(embeddings_test))  # 0=train, 1=test

print(f"\n🔄 Running t-SNE for all classes ({len(X)} samples)...")

# Giảm chiều bằng t-SNE
tsne = TSNE(n_components=2, perplexity=100, random_state=4512)
X_2d = tsne.fit_transform(X)

# Tính centroid và độ lệch chuẩn của từng class
unique_labels = np.unique(y)
stats = {}
for label in unique_labels:
    points = X_2d[y == label]
    centroid = points.mean(axis=0)
    std_radius = np.sqrt(((points - centroid) ** 2).sum(axis=1).mean())  # std theo khoảng cách
    stats[label] = {"centroid": centroid, "std": std_radius}

# Vẽ riêng từng class
for i, label in enumerate(unique_labels):
    color = colors[i % len(colors)]
    class_name = class_mapping['idx_to_class'][int(label)]

    plt.figure(figsize=(10, 8))
    plt.title(f"{class_name} (label {label})", fontsize=14, fontweight='bold')

    # Plot train/test của class hiện tại
    idx_train = (y == label) & (domain == 0)
    idx_test = (y == label) & (domain == 1)

    plt.scatter(X_2d[idx_train, 0], X_2d[idx_train, 1],
                s=30, alpha=0.8, color=color, marker='o', label='Train')
    plt.scatter(X_2d[idx_test, 0], X_2d[idx_test, 1],
                s=50, alpha=0.9, color=color, marker='x', label='Test')

    # Vẽ vòng tròn của các class khác
    for other_label in unique_labels:
        if other_label == label:
            continue
        cx, cy = stats[other_label]["centroid"]
        r = stats[other_label]["std"]

        circle = Circle((cx, cy), r, color='gray', fill=False, linestyle='--', linewidth=1.2, alpha=0.6)
        plt.gca().add_patch(circle)
        plt.text(cx, cy, class_mapping['idx_to_class'][int(other_label)],
                 fontsize=8, color='gray', ha='center', va='center')

    plt.legend()
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_path = f"tsne_per_class/tsne_class_{label}_{class_name}.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

print("✅ Done! Saved per-class t-SNE plots in tsne_per_class/")
