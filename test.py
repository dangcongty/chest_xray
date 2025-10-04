from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_p22/weights/best.pt')



results = model('datasets/process/images/0ea4221d568ab487af7c433a3df6307e.png', 
                imgsz = 1280,
                device = 'cuda:1',
                conf = 0.05)

results[0].save('test.jpg')