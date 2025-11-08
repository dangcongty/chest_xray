from ultralytics import YOLO

model = YOLO('runs/detect/single_cls5/weights/best.pt')
model.val(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            imgsz = 640,
            mosaic = 0,
            batch=8,
            device = 'cuda:0')

