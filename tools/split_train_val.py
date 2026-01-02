import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


class YOLODatasetSplitter:
    """
    Split YOLO detection dataset with balanced:
    - image count
    - box count
    - class distribution
    """

    def __init__(self, val_ratio=0.2, random_seed=42, limit_bg=None):
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.limit_bg = limit_bg
        np.random.seed(random_seed)

    # --------------------------------------------------
    def parse_yolo_label(self, label_path):
        annotations = []
        if not label_path.exists():
            return annotations

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                cls = int(parts[0])
                bbox = list(map(float, parts[1:5]))
                bbox = np.array(bbox)

                if (bbox < 0).any() or (bbox > 1).any():
                    print(f"⚠️ Invalid bbox in {label_path}")
                    return None

                annotations.append([cls] + bbox.tolist())

        return annotations

    # --------------------------------------------------
    def load_dataset(self, images_dir, labels_dir):
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        dataset = {}
        bg_count = 0
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in image_exts:
                continue

            label_path = labels_dir / f"{img_path.stem}.txt"
            annotations = self.parse_yolo_label(label_path)

            if annotations is None:
                continue

            if len(annotations) == 0:
                if self.limit_bg is not None and bg_count >= self.limit_bg:
                    continue
                bg_count += 1

            dataset[img_path.name] = annotations

        print(f"✓ Loaded {len(dataset)} images")
        return dataset

    # --------------------------------------------------
    def stratified_split(self, dataset):
        images_with_obj = []
        bg_images = []

        for img, anns in dataset.items():
            if len(anns) == 0:
                bg_images.append(img)
            else:
                images_with_obj.append(img)

        # ---- split images with objects (box-aware) ----
        img_infos = []
        for img in images_with_obj:
            anns = dataset[img]
            img_infos.append({
                "name": img,
                "n_boxes": len(anns),
                "classes": [a[0] for a in anns]
            })

        # sort large box images first
        img_infos.sort(key=lambda x: x["n_boxes"], reverse=True)

        train, val = [], []
        train_boxes = 0
        val_boxes = 0
        train_cls = defaultdict(int)
        val_cls = defaultdict(int)

        total_boxes = sum(x["n_boxes"] for x in img_infos)
        target_val_boxes = total_boxes * self.val_ratio

        for info in img_infos:
            if val_boxes < target_val_boxes:
                val.append(info["name"])
                val_boxes += info["n_boxes"]
                for c in info["classes"]:
                    val_cls[c] += 1
            else:
                train.append(info["name"])
                train_boxes += info["n_boxes"]
                for c in info["classes"]:
                    train_cls[c] += 1

        # ---- split background images (image ratio) ----
        np.random.shuffle(bg_images)
        bg_val_count = int(len(bg_images) * self.val_ratio)

        val += bg_images[:bg_val_count]
        train += bg_images[bg_val_count:]

        return train, val, train_cls, val_cls


    # --------------------------------------------------
    def print_statistics(self, dataset, train, val, train_cls, val_cls):
        total = len(train) + len(val)
        train_boxes = sum(len(dataset[i]) for i in train)
        val_boxes = sum(len(dataset[i]) for i in val)

        print("\n" + "=" * 60)
        print("YOLO DATASET SPLIT STATISTICS")
        print("=" * 60)
        print(f"Images: {total}")
        print(f"  Train: {len(train)} ({len(train)/total*100:.1f}%)")
        print(f"  Val:   {len(val)} ({len(val)/total*100:.1f}%)")
        print(f"Boxes:")
        print(f"  Train: {train_boxes}")
        print(f"  Val:   {val_boxes}")

        print("\nClass distribution:")
        print(f"{'Class':<8} {'Train':<10} {'Val':<10} {'Total'}")
        for c in sorted(set(train_cls) | set(val_cls)):
            t = train_cls[c]
            v = val_cls[c]
            print(f"{c:<8} {t:<10} {v:<10} {t+v}")

    # --------------------------------------------------
    def split_and_organize(self, images_dir, labels_dir, output_dir, copy_files=True):
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)

        dataset = self.load_dataset(images_dir, labels_dir)
        if not dataset:
            print("❌ Dataset empty")
            return None, None

        train, val, train_cls, val_cls = self.stratified_split(dataset)
        self.print_statistics(dataset, train, val, train_cls, val_cls)

        # write txt
        output_dir.mkdir(parents=True, exist_ok=True)
        train_txt = output_dir / "train.txt"
        val_txt = output_dir / "val.txt"

        with open(train_txt, "w") as f:
            for n in train:
                f.write(str(images_dir / n) + "\n")

        with open(val_txt, "w") as f:
            for n in val:
                f.write(str(images_dir / n) + "\n")

        # copy files
        if copy_files:
            for split, names in [("train", train), ("val", val)]:
                (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

                for n in names:
                    shutil.copy(images_dir / n, output_dir / split / "images" / n)
                    lbl = labels_dir / f"{Path(n).stem}.txt"
                    if lbl.exists():
                        shutil.copy(lbl, output_dir / split / "labels" / lbl.name)

        print("\n🎉 Done!")
        return train, val


splitter = YOLODatasetSplitter(
    val_ratio=0.2,
    random_seed=42,
    limit_bg=0   # None nếu không muốn giới hạn background
)

splitter.split_and_organize(
    images_dir="datasets/process/images",
    labels_dir="datasets/process/labels",
    output_dir="datasets/process",
    copy_files=False   # True nếu muốn copy ảnh + label
)
