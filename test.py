from glob import glob

from ultralytics import YOLO

model = YOLO('runs/detect/train43/weights/best.pt')

img_path = 'datasets/process/images/0a072917005494298d153c01bbd8f689.png'
results = model(img_path, 
                imgsz = 640,
                device = 'cuda:1',
                conf = 0.05,
                verbose = False,
                save = True)