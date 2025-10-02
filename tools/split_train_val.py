import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


class YOLODatasetSplitter:
    """
    Chia tập dữ liệu YOLO detection sao cho đều về:
    - Phân bố classes
    - Số lượng boxes
    - Số lượng images
    
    Format YOLO: mỗi dòng trong file .txt là [class_id x_center y_center width height]
    """
    
    def __init__(self, val_ratio=0.2, random_seed=42):
        """
        Args:
            val_ratio: tỉ lệ validation (0.2 = 20%)
            random_seed: seed để reproducible
        """
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def parse_yolo_label(self, label_path):
        """
        Đọc file label YOLO format
        Returns: list of [class_id, x, y, w, h]
        """
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        annotations.append([class_id] + bbox)
        except FileNotFoundError:
            pass  # File không có annotations
        return annotations
    
    def load_dataset(self, images_dir, labels_dir):
        """
        Load toàn bộ dataset từ thư mục images và labels
        
        Args:
            images_dir: thư mục chứa ảnh
            labels_dir: thư mục chứa file labels .txt
        
        Returns:
            dict: {image_filename: [annotations]}
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        
        dataset = {}
        
        # Tìm tất cả ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                # Tìm file label tương ứng
                label_path = labels_dir / f"{img_path.stem}.txt"
                annotations = self.parse_yolo_label(label_path)
                dataset[img_path.name] = annotations
        
        print(f"✓ Đã load {len(dataset)} images từ {images_dir}")
        return dataset
    
    def analyze_dataset(self, dataset):
        """
        Phân tích dataset để biết phân bố classes và boxes
        """
        stats = {
            'total_images': len(dataset),
            'total_boxes': 0,
            'class_distribution': defaultdict(int),
            'boxes_per_image': [],
            'images_per_class': defaultdict(list)
        }
        
        for img_name, annotations in dataset.items():
            n_boxes = len(annotations)
            stats['total_boxes'] += n_boxes
            stats['boxes_per_image'].append(n_boxes)
            
            classes_in_img = set()
            for ann in annotations:
                class_id = ann[0]
                stats['class_distribution'][class_id] += 1
                classes_in_img.add(class_id)
            
            for class_id in classes_in_img:
                stats['images_per_class'][class_id].append(img_name)
        
        return stats
    
    def stratified_split(self, dataset):
        """
        Chia dữ liệu theo phương pháp stratified sampling
        đảm bảo phân bố đều các classes
        """
        # Tạo list các image với thông tin về classes
        image_info = []
        for img_name, annotations in dataset.items():
            classes = set([ann[0] for ann in annotations])
            n_boxes = len(annotations)
            image_info.append({
                'name': img_name,
                'classes': classes,
                'n_boxes': n_boxes
            })
        
        # Sắp xếp theo số classes và số boxes để phân bố đều
        image_info.sort(key=lambda x: (len(x['classes']), x['n_boxes']))
        
        # Stratified sampling
        train_imgs = []
        val_imgs = []
        
        # Track số lượng mỗi class đã được chia
        train_class_count = defaultdict(int)
        val_class_count = defaultdict(int)
        train_box_count = 0
        val_box_count = 0
        
        for img in image_info:
            img_annotations = dataset[img['name']]
            img_boxes = len(img_annotations)
            
            if len(train_imgs) == 0:
                # Image đầu tiên vào train
                train_imgs.append(img['name'])
                train_box_count += img_boxes
                for class_id in img['classes']:
                    train_class_count[class_id] += sum(
                        1 for ann in img_annotations if ann[0] == class_id
                    )
            else:
                # Tính tỉ lệ hiện tại
                current_val_ratio = len(val_imgs) / (len(train_imgs) + len(val_imgs))
                
                # Tính score để quyết định train hay val
                val_score = 0
                train_score = 0
                
                for class_id in img['classes']:
                    train_total = train_class_count[class_id]
                    val_total = val_class_count[class_id]
                    total = train_total + val_total
                    
                    if total > 0:
                        current_train_ratio = train_total / total
                        current_val_ratio_cls = val_total / total
                        
                        # Ưu tiên thêm vào tập nào đang thiếu class này
                        val_score += abs((1 - self.val_ratio) - current_train_ratio)
                        train_score += abs(self.val_ratio - current_val_ratio_cls)
                
                # Quyết định dựa trên tỉ lệ và score
                if current_val_ratio < self.val_ratio or val_score < train_score:
                    val_imgs.append(img['name'])
                    val_box_count += img_boxes
                    for ann in img_annotations:
                        val_class_count[ann[0]] += 1
                else:
                    train_imgs.append(img['name'])
                    train_box_count += img_boxes
                    for ann in img_annotations:
                        train_class_count[ann[0]] += 1
        
        return train_imgs, val_imgs, train_class_count, val_class_count
    
    def print_statistics(self, dataset, train_imgs, val_imgs, 
                        train_class_count, val_class_count):
        """
        In thống kê về cách chia dữ liệu
        """
        print("\n" + "="*70)
        print("THỐNG KÊ CHIA DỮ LIỆU YOLO")
        print("="*70)
        
        # Thống kê tổng quan
        total = len(train_imgs) + len(val_imgs)
        print(f"\n📊 Tổng số images: {total}")
        print(f"   Train: {len(train_imgs):>6} ({len(train_imgs)/total*100:>5.1f}%)")
        print(f"   Val:   {len(val_imgs):>6} ({len(val_imgs)/total*100:>5.1f}%)")
        
        # Thống kê boxes
        train_boxes = sum(len(dataset[img]) for img in train_imgs)
        val_boxes = sum(len(dataset[img]) for img in val_imgs)
        total_boxes = train_boxes + val_boxes
        
        print(f"\n📦 Tổng số boxes: {total_boxes}")
        print(f"   Train: {train_boxes:>6} ({train_boxes/total_boxes*100:>5.1f}%)")
        print(f"   Val:   {val_boxes:>6} ({val_boxes/total_boxes*100:>5.1f}%)")
        
        # Thống kê theo class
        print(f"\n📋 Phân bố theo classes:")
        print(f"{'Class':<10} {'Train':<20} {'Val':<20} {'Total':<10}")
        print("-" * 70)
        
        all_classes = sorted(set(train_class_count.keys()) | set(val_class_count.keys()))
        for cls in all_classes:
            train_cnt = train_class_count.get(cls, 0)
            val_cnt = val_class_count.get(cls, 0)
            total_cnt = train_cnt + val_cnt
            
            if total_cnt > 0:
                print(f"{cls:<10} {train_cnt:>6} ({train_cnt/total_cnt*100:>5.1f}%) "
                      f"    {val_cnt:>6} ({val_cnt/total_cnt*100:>5.1f}%) "
                      f"    {total_cnt:<10}")
    
    def split_and_organize(self, images_dir, labels_dir, output_dir='dataset_split',
                          copy_files=True):
        """
        Chia dữ liệu và tổ chức theo cấu trúc YOLO
        
        Args:
            images_dir: thư mục chứa ảnh gốc
            labels_dir: thư mục chứa labels gốc
            output_dir: thư mục output
            copy_files: True để copy files, False để chỉ tạo file lists
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        output_dir = Path(output_dir)
        
        # Load dataset
        dataset = self.load_dataset(images_dir, labels_dir)
        
        if len(dataset) == 0:
            print("❌ Không tìm thấy dữ liệu!")
            return None, None
        
        # Chia dữ liệu
        train_imgs, val_imgs, train_class_count, val_class_count = \
            self.stratified_split(dataset)
        
        # In thống kê
        self.print_statistics(dataset, train_imgs, val_imgs,
                            train_class_count, val_class_count)
        

        with open('datasets/process/train.txt', 'w') as f:
            for name in train_imgs:
                path = f'datasets/process/images/{name}'
                f.write(f'{path}\n')
        with open('datasets/process/val.txt', 'w') as f:
            for name in val_imgs:
                path = f'datasets/process/images/{name}'
                f.write(f'{path}\n')

        print(f"\n🎉 Hoàn tất!")
        
        return train_imgs, val_imgs


# Ví dụ sử dụng
if __name__ == "__main__":
    # Cách 1: Chỉ tạo file lists (không copy files)
    splitter = YOLODatasetSplitter(val_ratio=0.2, random_seed=42)
    train_imgs, val_imgs = splitter.split_and_organize(
        images_dir='datasets/process/images',
        labels_dir='datasets/process/labels',
        output_dir='dataset_split',
        copy_files=False  # Chỉ tạo train.txt và val.txt
    )
    
    # Cách 2: Copy files và tổ chức theo cấu trúc YOLO
    # train_imgs, val_imgs = splitter.split_and_organize(
    #     images_dir='path/to/images',
    #     labels_dir='path/to/labels',
    #     output_dir='dataset_split',
    #     copy_files=True  # Copy files vào thư mục train/val
    # )