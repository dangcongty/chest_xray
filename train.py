from ultralytics import YOLO

model = YOLO('yolo11n.yaml')
model.train(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            imgsz = 640,
            batch=64,
            device='cuda:0',
            mosaic = 0,
            scale = 0,
            fliplr = 0,
            mixup = 0)