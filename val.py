
from ultralytics import YOLO

model_path = 'runs/heatmap/heatmap_yolo11-hm.yaml/weights/best.pt'
model = YOLO(model_path)
model.val()