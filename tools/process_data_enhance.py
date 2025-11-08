import os
import sys
from collections import defaultdict

import numpy as np
import pydicom
from scipy.cluster.hierarchy import fcluster, linkage

# from process_data import get_imageV1, resize_and_pad_image
sys.path.append(os.getcwd())
from utils.classes import CLASS_COLORS_BGR, CLASSES


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
    

def xyxy2scale_xywh(b, target_size):
    x1, y1, x2, y2 = b
    x = ((x1 + x2)/2)/target_size
    y = ((y1 + y2)/2)/target_size
    w = (x2 - x1)/target_size
    h = (y2 - y1)/target_size
    return [x, y, w, h]

def merge_boxes_robust(boxes, rad_ids, iou_thresh=0.5, merge_strategy='union'):
    """
    Merge overlapping boxes with proper edge case handling.
    
    Args:
        boxes: np.array of shape (N, 4) with format [x_min, y_min, x_max, y_max]
        rad_ids: list of radiologist IDs corresponding to each box
        iou_thresh: IoU threshold for considering boxes as overlapping
        merge_strategy: 'union', 'average', 'intersection', or 'weighted_inverse'
    
    Returns:
        merged_boxes: list of merged boxes
        merged_rad_ids: list of radiologist IDs for each merged box
        confidence_scores: list of confidence scores based on number of radiologists
    """
    
    if len(boxes) == 0:
        return [], [], []
    
    if len(boxes) == 1:
        return [boxes[0].tolist()], [rad_ids[0]], [0.5]
    
    n = len(boxes)
    
    # Step 1: Calculate IoU matrix
    iou_matrix = calculate_iou_matrix(boxes)
    
    # Step 2: Build adjacency matrix (which boxes should be grouped together)
    adjacency = (iou_matrix >= iou_thresh).astype(int)
    np.fill_diagonal(adjacency, 0)  # Remove self-connections
    
    # Step 3: Use hierarchical clustering to handle transitive overlaps
    # E.g., A overlaps B, B overlaps C, but A doesn't overlap C directly
    groups = cluster_boxes(adjacency, n)
    
    # Step 4: Merge boxes within each group
    merged_boxes = []
    merged_rad_ids = []
    confidence_scores = []
    
    for group_indices in groups:
        if len(group_indices) == 1:
            idx = group_indices[0]
            merged_boxes.append(boxes[idx].tolist())
            merged_rad_ids.append(rad_ids[idx])
            confidence_scores.append(0.5)
        else:
            # Multiple boxes to merge
            group_boxes = boxes[group_indices]
            group_rads = [rad_ids[i] for i in group_indices]
            
            # Merge based on strategy
            merged_box = merge_box_group(group_boxes, merge_strategy)
            
            # Flatten rad_ids if they're already lists
            all_rads = []
            for r in group_rads:
                if isinstance(r, list):
                    all_rads.extend(r)
                else:
                    all_rads.append(r)
            
            # Calculate confidence based on number of unique radiologists
            unique_rads = len(set(all_rads))
            confidence = calculate_confidence(unique_rads)
            
            merged_boxes.append(merged_box)
            merged_rad_ids.append(all_rads)
            confidence_scores.append(confidence)
    
    return merged_boxes, merged_rad_ids, confidence_scores


def calculate_iou_matrix(boxes):
    """Calculate IoU matrix for all pairs of boxes."""
    n = len(boxes)
    
    # Expand dimensions for broadcasting
    boxes1 = boxes[:, None, :]  # (N, 1, 4)
    boxes2 = boxes[None, :, :]  # (1, N, 4)
    
    # Intersection coordinates
    x1_inter = np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
    y1_inter = np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
    x2_inter = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
    y2_inter = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
    
    # Intersection area
    inter_width = np.maximum(0, x2_inter - x1_inter)
    inter_height = np.maximum(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Box areas
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Union area
    area1 = boxes_area[:, None]  # (N, 1)
    area2 = boxes_area[None, :]  # (1, N)
    union_area = area1 + area2 - inter_area
    
    # IoU with numerical stability
    iou_matrix = inter_area / np.maximum(union_area, 1e-8)
    
    return iou_matrix


def cluster_boxes(adjacency, n):
    """
    Cluster boxes using connected components to handle transitive overlaps.
    
    Example: If A overlaps B, and B overlaps C, then A, B, C form one group
    even if A doesn't directly overlap C.
    """
    visited = [False] * n
    groups = []
    
    def dfs(node, current_group):
        """Depth-first search to find all connected boxes."""
        visited[node] = True
        current_group.append(node)
        
        # Visit all neighbors
        for neighbor in range(n):
            if adjacency[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, current_group)
    
    # Find all connected components
    for i in range(n):
        if not visited[i]:
            current_group = []
            dfs(i, current_group)
            groups.append(current_group)
    
    return groups


def merge_box_group(boxes, strategy='union'):
    """
    Merge a group of boxes using specified strategy.
    
    Strategies:
    - 'union': Take min/max to include all areas (conservative)
    - 'average': Simple average of coordinates
    - 'intersection': Take only overlapping region (strict)
    - 'weighted_inverse': Weight smaller boxes higher (assumes tight = confident)
    """
    
    if strategy == 'union':
        x_min = np.min(boxes[:, 0])
        y_min = np.min(boxes[:, 1])
        x_max = np.max(boxes[:, 2])
        y_max = np.max(boxes[:, 3])
    
    elif strategy == 'average':
        x_min = np.mean(boxes[:, 0])
        y_min = np.mean(boxes[:, 1])
        x_max = np.mean(boxes[:, 2])
        y_max = np.mean(boxes[:, 3])
    
    elif strategy == 'intersection':
        x_min = np.max(boxes[:, 0])
        y_min = np.max(boxes[:, 1])
        x_max = np.min(boxes[:, 2])
        y_max = np.min(boxes[:, 3])
        
        # Handle edge case: no intersection
        if x_max <= x_min or y_max <= y_min:
            # Fall back to union if no intersection exists
            x_min = np.min(boxes[:, 0])
            y_min = np.min(boxes[:, 1])
            x_max = np.max(boxes[:, 2])
            y_max = np.max(boxes[:, 3])
    
    elif strategy == 'weighted_inverse':
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Inverse weighting (smaller boxes get higher weight)
        weights = 1.0 / (areas + 1e-8)
        weights = weights / np.sum(weights)
        
        x_min = np.sum(boxes[:, 0] * weights)
        y_min = np.sum(boxes[:, 1] * weights)
        x_max = np.sum(boxes[:, 2] * weights)
        y_max = np.sum(boxes[:, 3] * weights)
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")
    
    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def calculate_confidence(num_radiologists):
    """
    Calculate confidence score based on number of radiologists.
    
    You can customize this mapping based on your needs.
    """
    confidence_map = {
        1: 0.5,   # Only 1 radiologist
        2: 0.7,   # 2 radiologists agree
        3: 0.9,   # 3 radiologists agree
    }
    
    # For 4+ radiologists, cap at 0.95
    return confidence_map.get(num_radiologists, 0.95)


# ============================================================================
# Integration with your existing code
# ============================================================================

def process_enhanced(target_size, rad_scores, iou_thresh=0.5, merge_strategy='union'):
    """
    Enhanced version of your process function with robust edge case handling.
    """
    import os

    import cv2
    import pandas as pd
    from tqdm import tqdm
    
    annotations = pd.read_csv('datasets/annotations_train.csv')
    group_imageIDs = annotations.groupby('image_id')
    
    save_dir = 'datasets/process'
    os.makedirs(f'{save_dir}/images', exist_ok=True)
    os.makedirs(f'{save_dir}/labels_with_score', exist_ok=True)
    os.makedirs(f'{save_dir}/labels', exist_ok=True)
    os.makedirs(f'{save_dir}/vis', exist_ok=True)
    
    for image_id, annos in tqdm(group_imageIDs):
        has_finding = 'No finding' not in list(annos['class_name'])
        
        # Process image
        if not os.path.exists(f'{save_dir}/images/{image_id}.png'):
            img = get_imageV1(f'datasets/train/{image_id}.dicom')
            img, scale, pad_x, pad_y = resize_and_pad_image(img, target_size, pad_value=0)
            cv2.imwrite(f'{save_dir}/images/{image_id}.png', img)
        else:
            dcm = pydicom.dcmread(f'datasets/train/{image_id}.dicom')
            img = dcm.pixel_array.astype(np.float32)
            h, w = img.shape[:2]
            scale = target_size/max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            pad_x = (target_size - new_w) // 2
            pad_y = (target_size - new_h) // 2
            img = cv2.imread(f'{save_dir}/images/{image_id}.png')

        # Handle no finding case
        if not has_finding:
            with open(f'{save_dir}/labels/{image_id}.txt', 'w') as f:
                f.write("")
            with open(f'{save_dir}/labels_with_score/{image_id}.txt', 'w') as f:
                f.write("")
            continue
        
        # Process each class separately
        stores = {
            'radID': [],
            'classID': [],
            'boxes': [],
            'scores': []
        }
        
        group_class = annos.groupby('class_name')
        
        for class_name, gr in group_class:
            if class_name == 'No finding':
                continue
            
            # Extract boxes and rad_ids
            boxes = np.array([
                list(gr['x_min']), 
                list(gr['y_min']), 
                list(gr['x_max']), 
                list(gr['y_max'])
            ]).T
            
            rad_ids = list(gr['rad_id'])
            
            # Use robust merging
            merged_boxes, merged_rads, scores = merge_boxes_robust(
                boxes, 
                rad_ids, 
                iou_thresh=iou_thresh,
                merge_strategy=merge_strategy
            )
            
            # Store results
            for box, rads, score in zip(merged_boxes, merged_rads, scores):
                stores['boxes'].append(box)
                stores['classID'].append(class_name)
                stores['radID'].append(rads)
                stores['scores'].append(score)
        
        # Write labels to files
        with open(f'{save_dir}/labels_with_score/{image_id}.txt', 'w') as f_score:
            with open(f'{save_dir}/labels/{image_id}.txt', 'w') as f_raw:
                for box, class_name, rads, score in zip(
                    stores['boxes'], 
                    stores['classID'], 
                    stores['radID'], 
                    stores['scores']
                ):
                    if class_name not in CLASSES:
                        continue
                    
                    # Transform bbox
                    box = transform_bboxes(box, scale, pad_x, pad_y)
                    xywh = xyxy2scale_xywh(box, target_size)
                    
                    class_id = CLASSES.index(class_name)
                    
                    # Write with score
                    f_score.write(f'{class_id} {score:.4f} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n')
                    
                    # Write without score (for training)
                    f_raw.write(f'{class_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n')
        
        # Visualization (optional - can be commented out for speed)
        visualize_results(
            img, annos, stores, scale, pad_x, pad_y, 
            image_id, save_dir
        )


def visualize_results(img, annos, stores, scale, pad_x, pad_y, image_id, save_dir):
    """Visualize before/after merging."""
    import cv2

    # Before merging
    before = img.copy()
    for _, row in annos.iterrows():
        if row['class_name'] not in CLASSES:
            continue
        
        box = [row['x_min'], row['y_min'], row['x_max'], row['y_max']]
        box = transform_bboxes(box, scale, pad_x, pad_y)
        
        color = CLASS_COLORS_BGR[row['class_name']]
        before = cv2.rectangle(before, (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), color, 2)
        before = cv2.putText(before, f"{row['rad_id'][1:]}", 
                            (int(box[0])+10, int(box[1])+20), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    
    # After merging
    after = img.copy()
    for box, class_name, rads in zip(stores['boxes'], stores['classID'], stores['radID']):
        if class_name not in CLASSES:
            continue
        
        box = transform_bboxes(box, scale, pad_x, pad_y)
        color = CLASS_COLORS_BGR[class_name]
        
        after = cv2.rectangle(after, (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), color, 2)
        
        # Create label
        if isinstance(rads, str):
            label = rads[1:]
        else:
            label = f"{len(set(rads))}rads"
        
        after = cv2.putText(after, label, 
                           (int(box[0])+10, int(box[1])+20), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    
    # Combine and save
    vis = np.hstack([before, after])
    vis = cv2.putText(vis, image_id, (50, 50), 
                     cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 2)
    cv2.imwrite(f'{save_dir}/vis/{image_id}.jpg', vis)


# Example usage:
if __name__ == "__main__":
    
    rad_ids = ['R1', 'R2', 'R3', 'R4']
    
    print("Testing different merge strategies:\n")
    
    # for strategy in ['union', 'average', 'intersection', 'weighted_inverse']:
    strategy = 'average'
    # print(f"\nStrategy: {strategy}")
    # merged, rads, scores = merge_boxes_robust(boxes, rad_ids, iou_thresh=0.3, merge_strategy=strategy)
    
    # for i, (box, rad_list, score) in enumerate(zip(merged, rads, scores)):
    #     print(f"  Group {i+1}: Box={[int(x) for x in box]}, Rads={rad_list}, Score={score:.2f}")
    rad_scores = {
        1: 0.5,
        2: 0.75,
        3: 1,
    }
    process_enhanced(target_size=1280, 
                     rad_scores=rad_scores,
                     merge_strategy=strategy,
                     iou_thresh=0.1)