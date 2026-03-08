"""
Dataset Analyzer: So sánh phân phối Train vs Val (YOLO format)
Usage: python analyze_dataset.py --train train.txt --val val.txt
"""

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────
# 1. Đọc & parse dữ liệu
# ──────────────────────────────────────────────

def get_label_path(img_path: str) -> str:
    """Chuyển đường dẫn ảnh → đường dẫn label YOLO."""
    p = Path(img_path.strip())
    label = str(p).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    return str(Path(label).with_suffix(".txt"))


def parse_split(txt_file: str):
    """
    Đọc file .txt chứa danh sách ảnh, trả về:
      - images        : list đường dẫn ảnh
      - class_counts  : Counter {class_id: count}
      - boxes_per_img : list số bbox mỗi ảnh
      - wh_list       : list (w, h) relative của từng bbox
      - missing_labels: số file label không tìm thấy
    """
    images = []
    class_counts = Counter()
    boxes_per_img = []
    wh_list = []
    missing_labels = 0

    with open(txt_file, "r") as f:
        for line in f:
            img_path = line.strip()
            if not img_path:
                continue
            images.append(img_path)

            label_path = get_label_path(img_path)
            if not os.path.exists(label_path):
                missing_labels += 1
                boxes_per_img.append(0)
                continue

            boxes = 0
            with open(label_path, "r") as lf:
                for row in lf:
                    parts = row.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    class_counts[cls] += 1
                    wh_list.append((w, h))
                    boxes += 1
            boxes_per_img.append(boxes)

    return images, class_counts, boxes_per_img, wh_list, missing_labels


# ──────────────────────────────────────────────
# 2. Vẽ đồ thị
# ──────────────────────────────────────────────

def plot_analysis(train_data, val_data, class_names=None):
    (tr_imgs, tr_cls, tr_bpi, tr_wh, tr_miss) = train_data
    (va_imgs, va_cls, va_bpi, va_wh, va_miss) = val_data

    all_classes = sorted(set(list(tr_cls.keys()) + list(va_cls.keys())))
    if class_names is None:
        class_names = {c: f"Class {c}" for c in all_classes}

    labels = [class_names.get(c, f"Class {c}") for c in all_classes]
    tr_counts = [tr_cls.get(c, 0) for c in all_classes]
    va_counts = [va_cls.get(c, 0) for c in all_classes]

    # ── màu sắc ──
    C_TRAIN = "#4C9BE8"
    C_VAL   = "#F47C7C"
    C_BG    = "#0F1117"
    C_PANEL = "#1A1D27"
    C_TEXT  = "#E8EAF0"
    C_GRID  = "#2A2D3A"

    plt.rcParams.update({
        "figure.facecolor": C_BG,
        "axes.facecolor":   C_PANEL,
        "axes.edgecolor":   C_GRID,
        "axes.labelcolor":  C_TEXT,
        "xtick.color":      C_TEXT,
        "ytick.color":      C_TEXT,
        "text.color":       C_TEXT,
        "grid.color":       C_GRID,
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
        "font.family":      "DejaVu Sans",
    })

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Dataset Analysis  -  Train vs Val",
                 fontsize=18, fontweight="bold", color=C_TEXT, y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Tổng số ảnh ──
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(["Train", "Val"], [len(tr_imgs), len(va_imgs)],
                   color=[C_TRAIN, C_VAL], width=0.5, zorder=3)
    for bar, val in zip(bars, [len(tr_imgs), len(va_imgs)]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax1.set_title("Số lượng ảnh", fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.grid(axis="y", zorder=0)
    ratio = len(tr_imgs) / max(len(va_imgs), 1)
    ax1.set_xlabel(f"Train/Val ratio = {ratio:.1f}x", fontsize=9, color="#AAAAAA")

    # ── 2. Tổng số bbox ──
    ax2 = fig.add_subplot(gs[0, 1])
    tr_total = sum(tr_counts)
    va_total = sum(va_counts)
    bars2 = ax2.bar(["Train", "Val"], [tr_total, va_total],
                    color=[C_TRAIN, C_VAL], width=0.5, zorder=3)
    for bar, val in zip(bars2, [tr_total, va_total]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax2.set_title("Tổng số Bounding Boxes", fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.grid(axis="y", zorder=0)

    # ── 3. Avg bbox / ảnh ──
    ax3 = fig.add_subplot(gs[0, 2])
    tr_avg = np.mean(tr_bpi) if tr_bpi else 0
    va_avg = np.mean(va_bpi) if va_bpi else 0
    bars3 = ax3.bar(["Train", "Val"], [tr_avg, va_avg],
                    color=[C_TRAIN, C_VAL], width=0.5, zorder=3)
    for bar, val in zip(bars3, [tr_avg, va_avg]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax3.set_title("Avg Boxes / Ảnh", fontweight="bold")
    ax3.set_ylabel("Avg count")
    ax3.grid(axis="y", zorder=0)

    # ── 4. Phân phối class (grouped bar) ──
    ax4 = fig.add_subplot(gs[1, :])
    x = np.arange(len(labels))
    width = 0.38
    b1 = ax4.bar(x - width / 2, tr_counts, width, label="Train", color=C_TRAIN, zorder=3)
    b2 = ax4.bar(x + width / 2, va_counts, width, label="Val",   color=C_VAL,   zorder=3)
    ax4.set_title("Phân phối theo Class", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax4.set_ylabel("Số bbox")
    ax4.legend(framealpha=0.2)
    ax4.grid(axis="y", zorder=0)
    # Annotate
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                     str(int(h)), ha="center", va="bottom", fontsize=7)

    # ── 5. Phân phối boxes/ảnh (histogram) ──
    ax5 = fig.add_subplot(gs[2, 0])
    max_boxes = max(max(tr_bpi, default=0), max(va_bpi, default=0))
    bins = range(0, max_boxes + 2)
    ax5.hist(tr_bpi, bins=bins, alpha=0.7, color=C_TRAIN, label="Train", zorder=3, density=True)
    ax5.hist(va_bpi, bins=bins, alpha=0.7, color=C_VAL,   label="Val",   zorder=3, density=True)
    ax5.set_title("Phân phối Boxes/Ảnh", fontweight="bold")
    ax5.set_xlabel("Số boxes")
    ax5.set_ylabel("Tỉ lệ ảnh")
    ax5.legend(framealpha=0.2)
    ax5.grid(axis="y", zorder=0)

    # ── 6. BBox width distribution ──
    ax6 = fig.add_subplot(gs[2, 1])
    tr_w = [w for w, h in tr_wh]
    va_w = [w for w, h in va_wh]
    ax6.hist(tr_w, bins=30, alpha=0.7, color=C_TRAIN, label="Train", zorder=3, density=True)
    ax6.hist(va_w, bins=30, alpha=0.7, color=C_VAL,   label="Val",   zorder=3, density=True)
    ax6.set_title("BBox Width (relative)", fontweight="bold")
    ax6.set_xlabel("Width")
    ax6.set_ylabel("Density")
    ax6.legend(framealpha=0.2)
    ax6.grid(axis="y", zorder=0)

    # ── 7. BBox height distribution ──
    ax7 = fig.add_subplot(gs[2, 2])
    tr_h = [h for w, h in tr_wh]
    va_h = [h for w, h in va_wh]
    ax7.hist(tr_h, bins=30, alpha=0.7, color=C_TRAIN, label="Train", zorder=3, density=True)
    ax7.hist(va_h, bins=30, alpha=0.7, color=C_VAL,   label="Val",   zorder=3, density=True)
    ax7.set_title("BBox Height (relative)", fontweight="bold")
    ax7.set_xlabel("Height")
    ax7.set_ylabel("Density")
    ax7.legend(framealpha=0.2)
    ax7.grid(axis="y", zorder=0)

    # ── Footer stats ──
    fig.text(0.01, 0.01,
             f"Train: {len(tr_imgs)} imgs | {tr_total} boxes | {tr_miss} missing labels   "
             f"Val: {len(va_imgs)} imgs | {va_total} boxes | {va_miss} missing labels",
             fontsize=8, color="#888888")

    plt.savefig("dataset_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=C_BG)
    print("Saved: dataset_analysis.png")
    plt.show()



# ──────────────────────────────────────────────
# 3. In sample co nhieu box
# ──────────────────────────────────────────────

def print_high_box_samples(train_data, val_data, threshold=20):
    tr_imgs, _, tr_bpi, _, _ = train_data
    va_imgs, _, va_bpi, _, _ = val_data

    tr_high = [(img, n) for img, n in zip(tr_imgs, tr_bpi) if n > threshold]
    va_high = [(img, n) for img, n in zip(va_imgs, va_bpi) if n > threshold]

    print(f"\n{'='*60}")
    print(f"Anh co nhieu hon {threshold} boxes")
    print(f"{'='*60}")

    print(f"\n[TRAIN] {len(tr_high)} anh:")
    if tr_high:
        for img, n in sorted(tr_high, key=lambda x: -x[1]):
            print(f"  {n:>4} boxes  |  {img}")
    else:
        print("  (khong co)")

    print(f"\n[VAL] {len(va_high)} anh:")
    if va_high:
        for img, n in sorted(va_high, key=lambda x: -x[1]):
            print(f"  {n:>4} boxes  |  {img}")
    else:
        print("  (khong co)")

    print(f"\nTong: Train={len(tr_high)}, Val={len(va_high)}")
    print(f"{'='*60}\n")

# ──────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Dataset Analyzer")
    parser.add_argument("--train", default="/media/ssd220/ty/xray/datasets/process_v2/train_v3.txt", help="Path to train.txt")
    parser.add_argument("--val",   default="/media/ssd220/ty/xray/datasets/process_v2/val_v3.txt",   help="Path to val.txt")
    parser.add_argument("--names", default="/media/ssd220/ty/xray/datasets/dataset.yaml",
                        help="Optional: path to .names/.yaml class file")
    args = parser.parse_args()

    # Load class names nếu có
    class_names = None
    if args.names and os.path.exists(args.names):
        ext = Path(args.names).suffix
        if ext in [".names", ".txt"]:
            with open(args.names) as f:
                names = [l.strip() for l in f if l.strip()]
            class_names = {i: n for i, n in enumerate(names)}
        elif ext in [".yaml", ".yml"]:
            import yaml
            with open(args.names) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names", [])
            class_names = {i: n for i, n in enumerate(names)}

    print(f"Parsing train: {args.train}")
    train_data = parse_split(args.train)
    print(f"   -> {len(train_data[0])} images, {sum(train_data[2])} total boxes, "
          f"{train_data[4]} missing labels")

    print(f"Parsing val:   {args.val}")
    val_data = parse_split(args.val)
    print(f"   -> {len(val_data[0])} images, {sum(val_data[2])} total boxes, "
          f"{val_data[4]} missing labels")

    print_high_box_samples(train_data, val_data, threshold=20)

    print("Plotting...")
    plot_analysis(train_data, val_data, class_names)


if __name__ == "__main__":
    main()