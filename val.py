from ultralytics import YOLO

model = YOLO('runs/detect/train23/weights/best.pt')
model.val(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            imgsz = 640,
            mosaic = 0,
            batch=64,
            device = 'cuda:1')
