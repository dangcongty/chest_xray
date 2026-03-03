


from ultralytics import YOLO

model_path = 'runs/heatmap/from_yolo11m_pretraine_1k_bg.pt2/weights/best.pt'
model = YOLO(model_path)
model.predict(source='datasets/aNhan/train/images/0ee815af6f6dc10b6e9cf697791ca809.png', save = True)