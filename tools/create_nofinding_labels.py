import json
import os
import sys
from glob import glob

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from tools.statistics_data import box_pos, box_size

sys.path.append(os.getcwd())
from utils.classes import CLASS_COLORS_BGR, CLASSES


def center_distance(class_centers, abn_center):
    """
    Tính khoảng cách Euclidean giữa mỗi điểm trong class_centers và abn_center.
    
    Args:
        class_centers: Tensor [N, 2]  - tọa độ (x, y) của các ROI
        abn_center:  Tensor [2]     - tọa độ (x, y) của 1 box trung tâm

    Returns:
        Tensor [N] - khoảng cách Euclidean cho từng ROI
    """
    # Tính khoảng cách Euclidean
    distances = np.sqrt(np.sum((class_centers - abn_center) ** 2, axis=1))
    return distances

def box_iou(boxes, box):
    """
    Tính IoU giữa nhiều boxes và 1 box duy nhất.
    boxes: np.ndarray [N, 4]  (x1, y1, x2, y2)
    box:   np.ndarray [4]     (x1, y1, x2, y2)
    Trả về: np.ndarray [N] - IoU cho từng box
    """

    # Tính tọa độ giao nhau
    inter_x1 = np.maximum(boxes[:, 0], box[0])
    inter_y1 = np.maximum(boxes[:, 1], box[1])
    inter_x2 = np.minimum(boxes[:, 2], box[2])
    inter_y2 = np.minimum(boxes[:, 3], box[3])

    # Chiều rộng và cao của vùng giao
    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    # Diện tích từng box và box gốc
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_box = (box[2] - box[0]) * (box[3] - box[1])

    # IoU
    iou = inter_area / (area_boxes + area_box - inter_area + 1e-6)
    return iou

def center2box(nf_box_w_mean, nf_box_h_mean, sorted_near_centers):
    near_nf_boxes = []
    for center in sorted_near_centers:
        nf_x1 = center[0]-nf_box_w_mean/2
        nf_y1 = center[1]-nf_box_h_mean/2
        nf_x2 = center[0]+nf_box_w_mean/2
        nf_y2 = center[1]+nf_box_h_mean/2
        near_nf_boxes.append([nf_x1, nf_y1, nf_x2, nf_y2])
    near_nf_boxes = np.stack(near_nf_boxes, 0)
    return near_nf_boxes


nf_boxes_w, nf_boxes_h = box_size(plot=False)
centers = box_pos(plot=False)
all_txt_paths = glob('datasets/process/labels/*')
for txt_path in glob('datasets/process/labels/*'):

    # Chọn ngẫu nhiên N bệnh nhân khác
    num_labels = 3
    choosen_labels = np.random.choice(all_txt_paths, size = (num_labels))
    


    with open(txt_path, 'r') as f:
        labels = f.readlines()
    
    nf_labels = []
    for idx, label in enumerate(labels):
        label = label.strip().split()
        c = int(label[0])
        if c in [0, 3]:
            continue
        abn_box = np.array(label[1:], dtype = float) # xywh
        abn_x, abn_y, abn_w, abn_h = abn_box
        abn_x1, abn_y1, abn_x2, abn_y2 = abn_x - abn_w/2, abn_y - abn_h/2, abn_x + abn_w/2, abn_y + abn_h/2
        abn_center = np.array([abn_x, abn_y])
        abn_box = np.array([abn_x1, abn_y1, abn_x2, abn_y2])
        threshold_dist = np.sqrt(np.sum((abn_center - np.array([abn_x1, abn_y1]))**2))

        if abn_w > 0.1 or abn_h > 0.1:
            continue
        
        # nofinding boxes
        num_nf_boxes = 3
        nf_box_w, nf_box_h = nf_boxes_w[c], nf_boxes_h[c]
        # nf_box_w_mean = np.mean(nf_box_w)
        # nf_box_h_mean = np.mean(nf_box_h)
        nf_box_w_mean = abn_w
        nf_box_h_mean = abn_h

        class_centers = np.array(centers[c])
        np.random.shuffle(class_centers)
        # random choice center to generate boxes
        # filter by distance
        dists = center_distance(class_centers, abn_center)
        near_dists = dists > threshold_dist
        filtered_dists = dists[near_dists]
        near_centers = class_centers[near_dists]
        sorted_indices = np.argsort(filtered_dists)
        sorted_near_centers = near_centers[sorted_indices]
        near_nf_boxes = center2box(nf_box_w_mean, nf_box_h_mean, sorted_near_centers)
        # filter by iou
        ious = box_iou(near_nf_boxes, abn_box)
        accepted_ious = ious < 1e-6    
        iou_nf_boxes = near_nf_boxes[accepted_ious][:num_nf_boxes]   

        if len(iou_nf_boxes) == 0:
            continue
        

        # <class_id><index> <x> <y> <w> <h>
        for nf_box in iou_nf_boxes:
            nf_box_x = (nf_box[0] + nf_box[2])/2
            nf_box_y = (nf_box[1] + nf_box[3])/2
            nf_box_w = nf_box[2] - nf_box[0]
            nf_box_h = nf_box[3] - nf_box[1]
            nf_labels.append(f'{c}{idx} {nf_box_x} {nf_box_y} {nf_box_w} {nf_box_h}')

        # visualize 
        vis = cv2.imread(f'datasets/process/images/{os.path.basename(txt_path)[:-4]}.png')
        for nf_box in iou_nf_boxes:
            vis_nf_x1 = int(1280*nf_box[0])
            vis_nf_y1 = int(1280*nf_box[1])
            vis_nf_x2 = int(1280*nf_box[2])
            vis_nf_y2 = int(1280*nf_box[3])

            vis_abn_x1 = int(1280*(abn_center[0]-abn_w/2))
            vis_abn_y1 = int(1280*(abn_center[1]-abn_h/2))
            vis_abn_x2 = int(1280*(abn_center[0]+abn_w/2))
            vis_abn_y2 = int(1280*(abn_center[1]+abn_h/2))

            vis = cv2.rectangle(vis, [vis_nf_x1, vis_nf_y1], [vis_nf_x2, vis_nf_y2], (0, 255, 0), 2)
            vis = cv2.rectangle(vis, [vis_abn_x1, vis_abn_y1], [vis_abn_x2, vis_abn_y2], (0, 0, 255), 2)
        print()

    with open(f'datasets/process/nofinding_labels/{os.path.basename(txt_path)}', 'w') as f:
        if len(nf_labels):
            for nf_label in nf_labels:
                f.write(nf_label)
        else:
            f.write(f'')
