import glob
import os

import cv2
import numpy as np


def iou(box1, box2):
    # box = [x, y, w, h] (normalized)
    x1_min, x1_max = box1[0] - box1[2]/2, box1[0] + box1[2]/2
    y1_min, y1_max = box1[1] - box1[3]/2, box1[1] + box1[3]/2
    x2_min, x2_max = box2[0] - box2[2]/2, box2[0] + box2[2]/2
    y2_min, y2_max = box2[1] - box2[3]/2, box2[1] + box2[3]/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

# ---- CONFIG ----
mode = 'trainset'
if mode == 'trainset':
    LABEL_DIR = "datasets/process/labels_with_score"   # 📁 Thay bằng thư mục chứa .txt YOLO
    THRESH_IOU = 0.3
    SCORES = [0.75, 1.0]

    # ---- MAIN ----
    result = {}

    for label_path in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        boxes = []
        for line in lines:
            parts = list(map(float, line.split()))
            cls, score, x, y, w, h = parts
            boxes.append((int(cls), score, x, y, w, h))

        isolated_boxes = []
        for i, (cls_i, score_i, *box_i) in enumerate(boxes):
            if score_i < 0.7:
                continue
            overlapped = False
            for j, (cls_j, score_j, *box_j) in enumerate(boxes):
                if i == j or cls_i == cls_j:
                    continue
                if iou(box_i, box_j) >= THRESH_IOU:
                    overlapped = True
                    break
            if not overlapped:
                isolated_boxes.append({
                    "class": cls_i,
                    "score": score_i,
                    "box": box_i
                })

        if isolated_boxes:
            result[os.path.basename(label_path)] = isolated_boxes

    # ---- OUTPUT ----
    import json
    with open('datasets/non_overlap.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nTổng cộng: {len(result)} file có box đủ tiêu chí.")



    # crop box
    from tqdm import tqdm
    count = 0
    for img_id in tqdm(result):
        for rs in result[img_id]:
            try:
                c = int(rs['class'])
                box = np.array(rs['box'])

                os.makedirs(f'datasets/classify/{c}', exist_ok=True)
                img_path = f'datasets/process/images/{img_id[:-4]}.png'

                pad = 10 # 10 pixel
                img = cv2.imread(img_path)
                imgh, imgw = img.shape[:2]

                x, y, w, h = (box*1280).astype(int)
                x1 = x - w//2 
                x2 = x + w//2 
                y1 = y - h//2 
                y2 = y + h//2

                x1 = max(0, x1 - pad) 
                y1 = max(0, y1 - pad) 
                x2 = min(imgw, x2 + pad)
                y2 = min(imgh, y2 + pad)



                crop = img[y1:y2, x1:x2]
                cv2.imwrite(f'datasets/classify/{c}/{count}_{img_id[:-4]}.png', crop)
                count += 1
            except Exception as e:
                print(e)
                continue
else:
    LABEL_DIR = "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/Ty/vindr_cxr/labels"   # 📁 Thay bằng thư mục chứa .txt YOLO

    # ---- MAIN ----
    result = {}

    for label_path in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        boxes = []
        for line in lines:
            parts = list(map(float, line.split()))
            cls, x, y, w, h = parts
            boxes.append({
                "class": cls,
                "box": [x, y, w, h]
            })

        if len(boxes):
            result[os.path.basename(label_path)] = boxes

    # ---- OUTPUT ----
    import json
    with open('datasets/non_overlap_test.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nTổng cộng: {len(result)} file có box đủ tiêu chí.")



    # crop box
    from tqdm import tqdm
    count = 0
    for img_id in tqdm(result):
        for rs in result[img_id]:
            try:
                c = int(rs['class'])
                box = np.array(rs['box'])

                os.makedirs(f'datasets/classify/testset/{c}', exist_ok=True)
                img_path = f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/Ty/vindr_cxr/images/{img_id[:-4]}.png'

                pad = 10 # 10 pixel
                img = cv2.imread(img_path)
                imgh, imgw = img.shape[:2]

                x, y, w, h = (box*1280).astype(int)
                x1 = x - w//2 
                x2 = x + w//2 
                y1 = y - h//2 
                y2 = y + h//2

                x1 = max(0, x1 - pad) 
                y1 = max(0, y1 - pad) 
                x2 = min(imgw, x2 + pad)
                y2 = min(imgh, y2 + pad)



                crop = img[y1:y2, x1:x2]
                cv2.imwrite(f'datasets/classify/testset/{c}/{count}_{img_id[:-4]}.png', crop)
                count += 1
            except Exception as e:
                print(e)
                continue