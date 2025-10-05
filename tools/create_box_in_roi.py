import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statitics_data import box_pos, box_size

sys.path.append(os.getcwd())
from utils.classes import CLASS_COLORS_BGR, CLASSES


def bbox_intersects_polygon(bbox, polygon):
    """
    Kiểm tra xem bounding box (x_min, y_min, x_max, y_max)
    có cắt polygon hay không.
    
    bbox: list hoặc tuple [x_min, y_min, x_max, y_max]
    polygon: ndarray (N, 2) hoặc list các điểm [(x,y),...]
    """
    polygon = np.asarray(polygon)
    x_min, y_min, x_max, y_max = bbox

    # --- 1️⃣ Kiểm tra xem điểm polygon có nằm trong bbox không
    inside_x = (polygon[:, 0] >= x_min) & (polygon[:, 0] <= x_max)
    inside_y = (polygon[:, 1] >= y_min) & (polygon[:, 1] <= y_max)
    if np.any(inside_x & inside_y):
        return True

    # --- 2️⃣ Tạo các điểm của bbox
    bbox_pts = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])

    # --- 3️⃣ Kiểm tra xem đỉnh bbox có nằm trong polygon không
    if np.any(point_in_polygon(bbox_pts, polygon)):
        return True

    # --- 4️⃣ Kiểm tra giao nhau giữa cạnh bbox và cạnh polygon
    bbox_edges = list(zip(bbox_pts, np.roll(bbox_pts, -1, axis=0)))
    poly_edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))

    for a1, a2 in bbox_edges:
        for b1, b2 in poly_edges:
            if segments_intersect(a1, a2, b1, b2):
                return True

    return False


def bbox_overlap_ratio(bbox, polygon, samples=100):
    """
    Tính tỷ lệ diện tích của bbox nằm trong polygon (% overlap).
    Dùng grid sampling (ước lượng chính xác cao khi samples đủ lớn).
    
    bbox: [x_min, y_min, x_max, y_max]
    polygon: (N, 2)
    samples: số điểm chia mỗi chiều (samples^2 điểm)
    """
    polygon = np.asarray(polygon)
    x_min, y_min, x_max, y_max = bbox

    # tạo lưới điểm trong bbox
    xs = np.linspace(x_min, x_max, samples)
    ys = np.linspace(y_min, y_max, samples)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # kiểm tra điểm nằm trong polygon
    inside = point_in_polygon(pts, polygon)
    ratio = inside.mean()  # phần trăm điểm trong polygon ≈ phần trăm diện tích

    return ratio  # giá trị từ 0 → 1

# --------------------------------------
# 🔧 Hàm phụ: kiểm tra giao nhau giữa 2 đoạn thẳng
def segments_intersect(p1, p2, q1, q2):
    """Kiểm tra xem đoạn p1-p2 và q1-q2 có cắt nhau không"""
    def ccw(a, b, c):
        return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


# --------------------------------------
# 🔧 Hàm phụ: kiểm tra nhiều điểm có nằm trong polygon không (ray casting)
def point_in_polygon(points, polygon):
    """
    Ray-casting algorithm
    Trả về mảng bool: True nếu điểm nằm trong polygon
    """
    x = points[:, 0]
    y = points[:, 1]
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)

    for i in range(n):
        j = (i - 1) % n
        cond = ((poly_y[i] > y) != (poly_y[j] > y)) & \
               (x < (poly_x[j] - poly_x[i]) * (y - poly_y[i]) / (poly_y[j] - poly_y[i] + 1e-12) + poly_x[i])
        inside ^= cond  # XOR cho mỗi lần cắt
    return inside


def iou(box1, box2):
    """Tính IoU (Intersection over Union) giữa 2 bounding boxes"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)


def random_bboxes_in_polygon(polygon, w, h, n_boxes=10, max_tries=5000):
    """
    Sinh ngẫu nhiên các bounding box (x_min, y_min, x_max, y_max)
    nằm trong polygon và không chồng lên nhau (IoU = 0).
    """
    polygon = np.asarray(polygon).reshape((-1, 2))
    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    
    boxes = []
    tries = 0

    while len(boxes) < n_boxes and tries < max_tries:
        tries += 1

        center_x = np.random.uniform(min_x, max_x)
        center_y = np.random.uniform(min_y, max_y)

        x_min = center_x - w/2
        y_min = center_y - h/2
        x_max = center_x + w/2
        y_max = center_y + h/2
        new_box = [x_min, y_min, x_max, y_max]

        # 4 góc box
        corners = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ])

        # kiểm tra có nằm trong polygon không
        if bbox_overlap_ratio(new_box, polygon, samples=200) < 0.2:
            continue

        # kiểm tra IoU với các box đã có
        if all(iou(new_box, b) < 0.2 for b in boxes):
            boxes.append(new_box)
    
    return np.array(boxes)
if __name__ == '__main__':
    import json

    import cv2
    
    with open('datasets/object_rois.json', 'r') as f:
        object_rois = json.load(f)
    

    for c in range(14):
        if c in [9]:
            continue
        roi = object_rois[str(c)]
        box_w, box_h, polygons = roi
        vis = cv2.imread(f'datasets/vis_rois/{c}.jpg')

        all_boxes = []
        for poly in polygons:
            boxes = random_bboxes_in_polygon(poly, box_w, box_h, n_boxes=20, max_tries=1000)
            all_boxes.append(boxes)


            boxes_vis = (boxes.copy()*640).astype(int)
            for (x1, y1, x2, y2) in boxes_vis:
                vis = cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

        all_boxes = np.concatenate(all_boxes, 0)
        cv2.imwrite(f'datasets/vis_rois/{c}_box.jpg', vis)
        np.save(f'datasets/rois/{c}.npy', all_boxes)
