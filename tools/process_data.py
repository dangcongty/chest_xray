import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.classes import CLASS_COLORS_BGR, CLASSES


def resize_and_pad_image(img, target_size=1280, pad_value=0.0):
    """
    Resize và padding ảnh về (target_size, target_size), giữ nguyên tỷ lệ.
    Trả về ảnh mới, scale, pad_x, pad_y để dùng cho biến đổi bbox.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    # resize ảnh
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    # tạo ảnh vuông với padding
    new_img = np.full((target_size, target_size, 3), pad_value, dtype=img.dtype)

    # tính offset (padding trái, trên)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    # gắn ảnh vào giữa
    new_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img

    return new_img, scale, pad_x, pad_y

def get_pad_scale():
    return 

def transform_bboxes(box, scale, pad_x, pad_y):
    """
    Biến đổi bounding boxes theo scale và padding.
    Args:
        bboxes (list): list bbox [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = box
    x_min = int(x_min * scale + pad_x)
    y_min = int(y_min * scale + pad_y)
    x_max = int(x_max * scale + pad_x)
    y_max = int(y_max * scale + pad_y)
    return [x_min, y_min, x_max, y_max]

def get_imageV1(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def get_imageV2(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    raw_pixel = dcm.pixel_array
    img = cv2.convertScaleAbs(raw_pixel, alpha=(255.0/dcm.pixel_array.max()))
    if len(raw_pixel.shape) == 2:
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            img = cv2.bitwise_not(img)   
        img = cv2.equalizeHist(img)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = np.array(img)
    return img

def get_imageV3(dicom_path):
    """
    Đọc DICOM X-ray ngực và xử lý tối ưu cho phát hiện bệnh phổi
    """
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)
    
    # Xử lý multi-frame
    if len(img.shape) == 3:
        img = img[0]
    
    # Xử lý MONOCHROME1 (ảnh âm bản)
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img
    
    # ========== WINDOW/LEVEL (Quan trọng cho X-ray) ==========
    # X-ray chest thường: center ~40, width ~400
    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        center = float(dcm.WindowCenter)
        width = float(dcm.WindowWidth)
    else:
        # Default cho X-ray chest nếu không có thông tin
        center = 40
        width = 400
    
    img = np.clip(img, center - width/2, center + width/2)
    
    # Normalize về [0, 255]
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = np.zeros_like(img)
    
    img = img.astype(np.uint8)
    
    # ========== CLAHE (tốt hơn histogram equalization) ==========
    # Tăng contrast mà không làm mất chi tiết nhỏ (nodule, infiltrate, etc.)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Chuyển sang RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img

def get_imageV4(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)

    # Apply RescaleSlope/Intercept if present
    slope = getattr(dcm, 'RescaleSlope', 1)
    intercept = getattr(dcm, 'RescaleIntercept', 0)
    img = img * slope + intercept

    # Handle multi-frame
    if len(img.shape) == 3:
        img = img[0]

    # Handle MONOCHROME1 (negative)
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    # Window/Level (X-ray default)
    def get_first(x):
        return x[0] if isinstance(x, pydicom.multival.MultiValue) else x

    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        center = float(get_first(dcm.WindowCenter))
        width = float(get_first(dcm.WindowWidth))
    else:
        center, width = 40, 400  # Default for chest X-ray

    # Apply window and normalize
    img = np.clip((img - (center - width/2)) / width, 0, 1)
    img = img.astype(np.float32)

    meta = {
        'center': center,
        'width': width
    }
    return img, meta



def process_only_image(target_size):
    annotations = pd.read_csv('datasets/annotations_train.csv')
    group_imageIDs = annotations.groupby('image_id')
    
    save_dir = 'datasets/process'
    os.makedirs(f'{save_dir}/images', exist_ok=True)
    metadata = {}
    for image_id, annos in tqdm(group_imageIDs):
        # if '0c31081e8ada2990bcbef0f12ea60b07' not in image_id:
        #     continue
        # img, meta = get_imageV4(f'datasets/train/{image_id}.dicom')
        img = get_imageV1(f'datasets/train/{image_id}.dicom')
        img, scale, pad_x, pad_y = resize_and_pad_image(img, target_size, pad_value=0.0)
        # np.savez_compressed(f'{save_dir}/images/{image_id}.npz', img)
        # metadata[image_id] = meta
        cv2.imwrite(f'{save_dir}/images/{image_id}.png', img)

    # with open('metadata.json', 'w') as f:
    #     json.dump(metadata, f)


def process_test_set(target_size):
    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

    annotations = pd.read_csv('datasets/annotations_test.csv')
    annotations["class_index"] = annotations["class_name"].map(class_to_idx)

    group_imageIDs = annotations.groupby('image_id')
    
    save_dir = '/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/Ty/vindr_cxr'
    os.makedirs(f'{save_dir}/images', exist_ok=True)
    os.makedirs(f'{save_dir}/labels_with_score', exist_ok=True)
    os.makedirs(f'{save_dir}/labels', exist_ok=True)
    os.makedirs(f'{save_dir}/vis', exist_ok=True)
    for image_id, annos in tqdm(group_imageIDs):
        has_finding = 'No finding' not in list(annos['class_name'])
        # if not os.path.exists(f'{save_dir}/images/{image_id}.png'):
        img = get_imageV1(f'/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f42/Ty/vindr_cxr/test/{image_id}.dicom')
        imgh, imgw = img.shape[:2]
        img, scale, pad_x, pad_y = resize_and_pad_image(img, target_size, pad_value=0)
        cv2.imwrite(f'{save_dir}/images/{image_id}.png', img)
        vis = img.copy()

        if not has_finding:
            with open(f'{save_dir}/labels/{image_id}.txt', 'w') as f:
                f.write("")
        else:
            class_names = list(annos['class_name'])
            boxes = np.array([list(annos['x_min']), list(annos['y_min']), list(annos['x_max']), list(annos['y_max'])]).T
            gt_classes = np.array(list(annos['class_index'])).T
            with open(f'{save_dir}/labels/{image_id}.txt', 'w') as f:
                for c, box in zip(gt_classes, boxes):
                    if np.isnan(c):
                        # các class ít
                        continue
                    _box = transform_bboxes(box, scale, pad_x, pad_y)
                    __box = xyxy2scale_xywh(_box, target_size)

                    x, y, w, h = __box

                    f.write(f"{int(c)} {float(x)} {float(y)} {float(w)} {float(h)}\n")

                    vis = cv2.rectangle(vis, _box[:2], _box[2:], CLASS_COLORS_BGR[int(c)], 2)
                    vis = cv2.putText(vis, f'{CLASSES[int(c)]}', _box[:2], cv2.FONT_HERSHEY_DUPLEX, 0.7, CLASS_COLORS_BGR[int(c)], 1)

            cv2.imwrite(f'{save_dir}/vis/{image_id}.jpg', vis)


def process(target_size, rad_scores, iou_thresh):
    annotations = pd.read_csv('datasets/annotations_train.csv')
    group_imageIDs = annotations.groupby('image_id')
    
    save_dir = 'datasets/process'
    os.makedirs(f'{save_dir}/images', exist_ok=True)
    os.makedirs(f'{save_dir}/labels_with_score', exist_ok=True)
    os.makedirs(f'{save_dir}/labels', exist_ok=True)
    os.makedirs(f'{save_dir}/vis', exist_ok=True)
    for image_id, annos in tqdm(group_imageIDs):
        # if '3fd94dd5' not in image_id:
        #     continue
        has_finding = 'No finding' not in list(annos['class_name'])

        
        if not os.path.exists(f'{save_dir}/images/{image_id}.png'):
            img = get_imageV1(f'datasets/train/{image_id}.dicom')
            img, scale, pad_x, pad_y = resize_and_pad_image(img, target_size, pad_value=0)
            cv2.imwrite(f'{save_dir}/images/{image_id}.png', img)
            # continue

        if not has_finding:
            with open(f'{save_dir}/labels/{image_id}.txt', 'w') as f:
                f.write("")
        
        else:
            stores = {
                'radID': [],
                'classID': [],
                'boxes': [],
                'scores': []
            }    
            rad_ids = list(annos['rad_id'])
            class_names = list(annos['class_name'])
            group_class = annos.groupby('class_name')
            for c, gr in group_class:
                if len(gr) == 1:
                    r = str(gr['rad_id'].iloc[0])
                    b = [float(gr['x_min'].iloc[0]), float(gr['y_min'].iloc[0]), float(gr['x_max'].iloc[0]), float(gr['y_max'].iloc[0])]
                    stores["radID"].append(r)
                    stores["classID"].append(c)
                    stores["boxes"].append(b)
                    stores["scores"].append(0.5)
                else:
                    rads = np.unique(gr['rad_id'])
                    all_rads = np.array(gr['rad_id'])
                    num_unique_rads = len(rads)

                    boxes = np.array([list(gr['x_min']), list(gr['y_min']), list(gr['x_max']), list(gr['y_max'])]).T
                    n = len(boxes)
                    # Expand dimensions
                    boxes1 = boxes[:, None, :]  # (N, 1, 4)
                    boxes2 = boxes[None, :, :]  # (1, N, 4)
                    # Intersection coordinates
                    x1_inter = np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
                    y1_inter = np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
                    x2_inter = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
                    y2_inter = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
                    # Area intersection
                    inter_width = np.maximum(0, x2_inter - x1_inter)
                    inter_height = np.maximum(0, y2_inter - y1_inter)
                    inter_area = inter_width * inter_height
                    # Area each box
                    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    # Area Union
                    area1 = boxes_area[:, None]  # (N, 1)
                    area2 = boxes_area[None, :]  # (1, N)
                    union_area = area1 + area2 - inter_area
                    # IoU
                    iou_matrix = inter_area / np.maximum(union_area, 1e-8)
                    mask_diagonal = 1 - np.eye(n)
                    iou_matrix = iou_matrix * mask_diagonal
                    count_overlaps = np.where(iou_matrix > iou_thresh, 1, 0)

                    already_merge = []
                    for b_id, box1 in enumerate(boxes):
                        merge_box = None
                        count = count_overlaps.sum(-1)[b_id]
                        if b_id not in already_merge:
                            if count > 0:
                                box_merge_idx = np.argwhere(count_overlaps[b_id] == 1)

                                for box2 in boxes[box_merge_idx].reshape((-1, 4)):
                                    if merge_box is None:
                                        x11, y11, x12, y12 = box1
                                    else:
                                        x11, y11, x12, y12 = merge_box
                                    
                                    x21, y21, x22, y22 = box2

                                    x1m = (x21 + x11)/2
                                    y1m = (y21 + y11)/2
                                    x2m = (x22 + x12)/2
                                    y2m = (y22 + y12)/2
                                    
                                    merge_box = [x1m, y1m, x2m, y2m]
                                merge_box = np.array(merge_box, dtype = np.int32).tolist()
                                merge_rad = all_rads[box_merge_idx].flatten().tolist() + [all_rads[b_id]]
                                stores["radID"].append(merge_rad)
                                stores["classID"].append(c)
                                stores["boxes"].append(merge_box)
                                stores["scores"].append(rad_scores[len(merge_rad)] if len(merge_rad) <= 3 else rad_scores[3])
                                already_merge += box_merge_idx.flatten().tolist()
                            else:
                                stores["radID"].append(list(gr['rad_id'])[b_id])
                                stores["classID"].append(c)
                                stores["boxes"].append(boxes[b_id])
                                stores["scores"].append(0.5)

            
            # write to file  
            box_after = np.array(stores['boxes']).astype(np.uint16)
            class_after = np.array(stores['classID'])
            rad_after = list(stores['radID'])
            scores = list(stores['scores'])
            
            with open(f'{save_dir}/labels_with_score/{image_id}.txt', 'w') as f:
                with open(f'{save_dir}/labels/{image_id}.txt', 'w') as fraw:
                    for b, c, r, s in zip(box_after, class_after, rad_after, scores):
                        b = transform_bboxes(b, scale, pad_x, pad_y)
                        xywh = xyxy2scale_xywh(b, target_size)
                        if c not in CLASSES:
                            continue
                        cid = CLASSES.index(c)
                        f.write(f'{cid} {s} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')
                        fraw.write(f'{cid} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')


            # visualize
            before = img.copy()
            xmin = np.array(annos['x_min'])
            ymin = np.array(annos['y_min'])
            xmax = np.array(annos['x_max'])
            ymax = np.array(annos['y_max'])
            box_before = np.array([xmin, ymin, xmax, ymax]).T.astype(np.uint16)
            class_before = list(annos['class_name'])
            rad_before = list(annos['rad_id'])
            for bb, cb, rb in zip(box_before, class_before, rad_before):
                bb = transform_bboxes(bb, scale, pad_x, pad_y)
                if cb not in CLASSES:
                    continue
                before = cv2.rectangle(before, bb[:2], bb[2:], CLASS_COLORS_BGR[cb], 2)
                before = cv2.putText(before, f'{rb[1:]}', (bb[0]+10, bb[1]+20), cv2.FONT_HERSHEY_DUPLEX, 0.7, CLASS_COLORS_BGR[cb], 1)

            
            after = img.copy()
            for ba, ca, ra in zip(box_after, class_after, rad_after):
                ba = transform_bboxes(ba, scale, pad_x, pad_y)
                if ca not in CLASSES:
                    continue
                after = cv2.rectangle(after, ba[:2], ba[2:], CLASS_COLORS_BGR[ca], 2)
                after = cv2.putText(after, f'{ra[1:] if isinstance(ra, str) else "_".join(ra)}', (ba[0]+10, ba[1]+20), cv2.FONT_HERSHEY_DUPLEX, 0.7, CLASS_COLORS_BGR[ca], 1)

            visualize = np.hstack([before, after])
            visualize = cv2.putText(visualize, image_id, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
            cv2.imwrite(f'{save_dir}/vis/{image_id}.jpg', visualize)
    

def xyxy2scale_xywh(b, target_size):
    x1, y1, x2, y2 = b
    x = ((x1 + x2)/2)/target_size
    y = ((y1 + y2)/2)/target_size
    w = (x2 - x1)/target_size
    h = (y2 - y1)/target_size
    return [x, y, w, h]


if __name__ == '__main__':
    rad_scores = {
        1: 0.5,
        2: 0.75,
        3: 1,
    }
    target_size = 1280
    iou_thresh = 0.1
    process(target_size, rad_scores, iou_thresh)

    # process_only_image(target_size)
    # process_test_set(target_size)