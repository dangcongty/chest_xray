from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            imgsz = 640,
            batch = 64,
            device = 'cuda:0',
            name = 'test',
            hsv_v = 0.1,
            mosaic = 0,
            scale = 0,
            fliplr = 0,
            mixup = 0,
            
            
            # contrastive
            use_contrastive = True,
            use_conf_aware = False
            )