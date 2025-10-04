from ultralytics import YOLO

# model = YOLO('ultralytics/cfg/models/11/yolo11-p2.yaml')
model = YOLO('runs/detect/yolo11_p2/weights/best.pt')
model.train(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            
            # hyp
            epochs = 500,
            imgsz = 1280,
            batch = 16,
            cos_lr = True,

            # loss
            box = 15,
            cls = 0.5,
            dfl = 1.5,


            # others
            device = 'cuda:0',
            name = 'yolo11_p2',
            plots = True,
            resume = True,

            # augmentation
            mosaic = 0,
            fliplr = 0,
            mixup = 0,
            hsv_v = 0.1,
            scale = 0.1, # mô phỏng vị trí gần - xa máy chụp
            degrees = 5,
            translate = 0.1,
            
            
            # contrastive
            use_contrastive = True,
            use_conf_aware = False
            )