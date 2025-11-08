import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Chuyển YOLO -> xyxy ---
def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w/2) * img_w
    y1 = (y_center - h/2) * img_h
    x2 = (x_center + w/2) * img_w
    y2 = (y_center + h/2) * img_h
    return x1, y1, x2, y2

# --- Tính IoU ---
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter_area / (area1 + area2 - inter_area)

# --- Đếm overlaps + số lượng box ---
def count_overlaps_in_yolo_folder(label_dir, img_w, img_h, iou_threshold=0.8):
    overlap_counts = defaultdict(lambda: defaultdict(int))
    box_counts = defaultdict(int)
    all_classes = set()

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        with open(os.path.join(label_dir, label_file)) as f:
            lines = [l.strip().split() for l in f.readlines() if l.strip()]

        boxes = []
        for line in lines:
            cls, x, y, w, h = map(float, line)
            cls = int(cls)
            all_classes.add(cls)
            box_counts[cls] += 1
            box = yolo_to_xyxy(x, y, w, h, img_w, img_h)
            boxes.append((cls, box))

        for (cls1, box1), (cls2, box2) in itertools.combinations(boxes, 2):
            if cls1 == cls2:
                continue
            if iou(box1, box2) > iou_threshold:
                overlap_counts[cls1][cls2] += 1
                overlap_counts[cls2][cls1] += 1  # để có raw đối xứng

    return overlap_counts, box_counts, sorted(all_classes)

# --- Xây raw + normalized matrix ---
def build_matrices(overlap_counts, box_counts, class_list):
    n = len(class_list)
    raw_matrix = np.zeros((n, n), dtype=float)
    norm_matrix = np.zeros((n, n), dtype=float)

    for i, a in enumerate(class_list):
        for j, b in enumerate(class_list):
            if a == b:
                continue
            count_ab = overlap_counts[a].get(b, 0)
            raw_matrix[i, j] = count_ab
            total_a = box_counts.get(a, 1)
            norm_matrix[i, j] = count_ab / total_a if total_a > 0 else 0

    return raw_matrix, norm_matrix

# --- Vẽ 2 heatmaps cạnh nhau ---
def plot_two_matrices(raw_matrix, norm_matrix, class_list, iou_threshold):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    titles = [
        f"Raw Overlap Count (IoU>{iou_threshold})",
        f"Normalized Overlap (Overlap / CountA)"
    ]
    cmaps = ["Blues", "YlGn"]

    for ax, matrix, title, cmap in zip(axs, [raw_matrix, norm_matrix], titles, cmaps):
        im = ax.imshow(matrix, cmap=cmap)
        n = len(class_list)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(class_list)
        ax.set_yticklabels(class_list)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.2f}" if title.startswith("Normalized") else int(val),
                            ha="center", va="center", color="black", fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Class Overlap Analysis", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig('datasets/overlap.png')

# --- Example usage ---
label_folder = "datasets/process/labels"
img_w, img_h = 640, 640
iou_threshold = 0.85

overlap_counts, box_counts, class_list = count_overlaps_in_yolo_folder(label_folder, img_w, img_h, iou_threshold)
raw_matrix, norm_matrix = build_matrices(overlap_counts, box_counts, class_list)
plot_two_matrices(raw_matrix, norm_matrix, class_list, iou_threshold)
