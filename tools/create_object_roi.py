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




boxes_w, boxes_h = box_size(plot=False)
centers = box_pos(plot=False)

object_rois = {}
for c in range(14):
    box_w, box_h = boxes_w[c], boxes_h[c]

    box_w_mean = np.mean(box_w)
    box_h_mean = np.mean(box_h)

    box_centers = centers[c]
    box_centers = (np.array(box_centers)*640).astype(int)
    
    clf = IsolationForest(contamination=0.05, random_state=0)
    inliers = clf.fit_predict(box_centers) == 1
    filtered_points = box_centers[inliers]
    polygons = []

    if c in [0, 3, 9]:
        hull = ConvexHull(filtered_points)
        polygon = filtered_points[hull.vertices]
        polygons.append(polygon)
    elif c == 11:
        kmeans = KMeans(n_clusters=4, random_state=0)
        labels = kmeans.fit_predict(filtered_points)
        # --- 2️⃣ Tạo convex hull riêng cho từng cụm ---
        for cluster_id in np.unique(labels):
            cluster_points = filtered_points[labels == cluster_id]
            if len(cluster_points) >= 3:  # cần ít nhất 3 điểm để tạo polygon
                hull = ConvexHull(cluster_points)
                polygons.append(cluster_points[hull.vertices])

    else:
        # --- 1️⃣ Phân cụm các điểm inlier thành 2 nhóm ---
        kmeans = KMeans(n_clusters=2, random_state=0)
        labels = kmeans.fit_predict(filtered_points)
        # --- 2️⃣ Tạo convex hull riêng cho từng cụm ---
        for cluster_id in np.unique(labels):
            cluster_points = filtered_points[labels == cluster_id]
            if len(cluster_points) >= 3:  # cần ít nhất 3 điểm để tạo polygon
                hull = ConvexHull(cluster_points)
                polygons.append(cluster_points[hull.vertices])


    vis = np.zeros((640, 640, 3), dtype=np.uint8)
    for x, y in box_centers:
        try:
            vis[y, x] = CLASS_COLORS_BGR[CLASSES[c]]
        except:
            continue
    for polygon in polygons:
        vis = cv2.polylines(vis, [polygon.reshape((-1, 1, 2)).astype(np.int32)], True, CLASS_COLORS_BGR[CLASSES[c]], 3)
    cv2.imwrite(f'datasets/vis_rois/{c}.jpg', vis)
    
    polygons = [(poly/640).tolist() for poly in polygons]
    object_rois[c] = [float(box_w_mean), float(box_h_mean), polygons]

with open('datasets/object_rois.json', 'w') as f:
    json.dump(object_rois, f)