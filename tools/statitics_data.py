from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def class_distribute():
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




def box_size():
    stats_w = {i: [] for i in range(14)}
    stats_h = {i: [] for i in range(14)}
    for txt_path in glob('datasets/process/labels/*'):
        with open(txt_path, 'r') as f:
            data = f.readlines()
    
        for dt in data:
            dt = dt.strip().split()
            c = int(dt[0])
            x_center, y_center, w, h = np.array(dt[1:], dtype=float)
            if w < 0 or h < 0:
                continue
            if w > 0.9 or h > 0.9:
                print(txt_path)

            stats_w[c].append(w)
            stats_h[c].append(h)
    
    labels = [f'{i}' for i in range(14)]

    boxsize_w = [stats_w[c] for c in stats_w]
    boxsize_h = [stats_h[c] for c in stats_h]

    fig, ax = plt.subplots(figsize=(20, 6))

    positions1 = [i - 0.2 for i in range(1, len(boxsize_w) + 1)]
    positions2 = [i + 0.2 for i in range(1, len(boxsize_h) + 1)]

    # Vẽ boxplot cho W
    bp1 = ax.boxplot(boxsize_w, positions=positions1, widths=0.3, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor("lightblue")

    # Vẽ boxplot cho H
    bp2 = ax.boxplot(boxsize_h, positions=positions2, widths=0.3, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor("lightgreen")

    # Hiển thị median
    for pos, data in zip(positions1, boxsize_w):
        median = np.median(data)
        ax.text(pos, median, f"{median:.1f}", ha='center', va='bottom', fontsize=9, color="blue")

    for pos, data in zip(positions2, boxsize_h):
        median = np.median(data)
        ax.text(pos, median, f"{median:.1f}", ha='center', va='bottom', fontsize=9, color="green")

    # Hiển thị mean (chấm đỏ)
    for pos, data in zip(positions1, boxsize_w):
        mean = np.mean(data)
        ax.plot(pos, mean, "ro", markersize=4)

    for pos, data in zip(positions2, boxsize_h):
        mean = np.mean(data)
        ax.plot(pos, mean, "ro", markersize=4)

    # Hiển thị số outlier / tổng - %
    for pos, data, fliers in zip(positions1, boxsize_w, bp1['fliers']):
        total = len(data)
        out = len(fliers.get_ydata())
        perc = out / total * 100
        y = np.max(data) * 1.05
        ax.text(pos, y, f"{out}/{total}\n{perc:.1f}%", 
                ha='center', va='bottom', fontsize=8, color="blue", rotation=45)

    for pos, data, fliers in zip(positions2, boxsize_h, bp2['fliers']):
        total = len(data)
        out = len(fliers.get_ydata())
        perc = out / total * 100
        y = np.max(data) * 1.05
        ax.text(pos, y, f"{out}/{total}\n{perc:.1f}%", 
                ha='center', va='bottom', fontsize=8, color="green", rotation=45)

    # Cấu hình trục
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Class", fontsize=14)
    ax.set_ylabel("px norm", fontsize=14)
    ax.set_title("So sánh kích thước bounding box giữa các classa", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Legend để dưới cùng, không che chữ
    legend_elements = [
        Patch(facecolor='lightblue', label='Boxsize W'),
        Patch(facecolor='lightgreen', label='Boxsize H'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()
    plt.savefig("datasets/box_size_chart.jpg", dpi=300)



def box_pos():
    stats_center = {i: [] for i in range(14)}
    for txt_path in glob('datasets/process/labels/*'):
        with open(txt_path, 'r') as f:
            data = f.readlines()
    
        for dt in data:
            dt = dt.strip().split()
            c = int(dt[0])
            x_center, y_center, w, h = np.array(dt[1:], dtype=float)
            if w < 0 or h < 0:
                continue
            if w > 0.9 or h > 0.9:
                print(txt_path)

            stats_center[c].append((x_center, y_center))
        
    # Lưu dữ liệu
    stats_center = defaultdict(list)

    # Đọc dữ liệu YOLO
    for txt_path in glob('datasets/process/labels/*.txt'):
        with open(txt_path, 'r') as f:
            data = f.readlines()
        for dt in data:
            dt = dt.strip().split()
            if len(dt) < 5:
                continue
            c = int(dt[0])
            x_center, y_center, w, h = np.array(dt[1:], dtype=float)
            if w < 0 or h < 0:
                continue
            if w > 0.9 or h > 0.9:
                print("Box quá lớn:", txt_path)
            stats_center[c].append((x_center, y_center))

        # Số class
        num_classes = max(stats_center.keys()) + 1
        colors = plt.cm.get_cmap("tab20", num_classes)

        # -------- Chart tổng (tất cả class) --------
        plt.figure(figsize=(15, 10))
        for c, coords in stats_center.items():
            coords = np.array(coords)
            if coords.size == 0:
                continue
            plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6,
                        color=colors(c), label=f"Class {c}")
        plt.xlabel("x_center", fontsize=14)
        plt.ylabel("y_center", fontsize=14)
        plt.title("Phân bố vị trí box - Tất cả class", fontsize=16, weight="bold")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, markerscale = 2)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("datasets/box_position_all_classes_chart.jpg", dpi=300, bbox_inches="tight")

        # -------- Chart riêng từng class --------
        cols = 4  # số cột trong subplot
        rows = int(np.ceil(num_classes / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten()

        for c in range(num_classes):
            ax = axes[c]
            coords = np.array(stats_center[c])
            if coords.size == 0:
                ax.set_visible(False)
                continue
            ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6, color=colors(c))
            ax.set_title(f"Class {c}", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)

        # Ẩn subplot thừa
        for i in range(num_classes, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Phân bố vị trí box theo từng class", fontsize=18, weight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("datasets/box_position_per_class_chart.jpg", dpi=300)

def box_co_occurrence():
    # Lưu class theo ảnh
    image_classes = []

    for txt_path in glob('datasets/process/labels/*.txt'):
        with open(txt_path, 'r') as f:
            data = f.readlines()
        classes = set()
        for dt in data:
            dt = dt.strip().split()
            if len(dt) < 5:
                continue
            c = int(dt[0])
            classes.add(c)
        if classes:
            image_classes.append(list(classes))


    # Tổng số class
    num_classes = max(max(cls) for cls in image_classes) + 1

    # -----------------------------
    # 2. Tạo ma trận đồng xuất hiện (đếm số ảnh có i và j)
    # -----------------------------
    co_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for classes in image_classes:
        for i in classes:
            for j in classes:
                if i != j:
                    co_matrix[i, j] += 1

    # -----------------------------
    # 3. Tính Jaccard Index: |A ∩ B| / |A ∪ B|
    # -----------------------------
    appear_count = np.zeros(num_classes, dtype=int)
    for classes in image_classes:
        for c in classes:
            appear_count[c] += 1

    jaccard = np.zeros_like(co_matrix, dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            denom = appear_count[i] + appear_count[j] - co_matrix[i, j]
            if denom > 0:
                jaccard[i, j] = co_matrix[i, j] / denom

    # -----------------------------
    # 4. Vẽ heatmap Jaccard
    # -----------------------------
    plt.figure(figsize=(10, 8))
    im = plt.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im)

    plt.title("Ma trận thể hiện khả năng các class đi chung với nhau", fontsize=16, weight="bold")
    plt.xlabel("Class j", fontsize=14)
    plt.ylabel("Class i", fontsize=14)

    # Tick labels
    plt.xticks(np.arange(num_classes), [f"C{c}" for c in range(num_classes)], rotation=90)
    plt.yticks(np.arange(num_classes), [f"C{c}" for c in range(num_classes)])

    # Hiển thị giá trị %
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and jaccard[i, j] > 0:
                plt.text(j, i, f"{jaccard[i,j]*100:.1f}%", 
                        ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    plt.savefig("datasets/co_occurrence_heatmap.jpg", dpi=300)


if __name__ == '__main__':
    # class_distribute()
    box_size()
