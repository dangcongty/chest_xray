from ultralytics import YOLO

# model = YOLO('ultralytics/cfg/models/11/yolo11n-p2.yaml')
model = YOLO('runs/detect/train/weights/last.pt')
model.train(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 8,
            cos_lr = True,

            # loss
            box = 7.5,
            # cls = 0.5,
            cls = 5,
            dfl = 1.5,

            # others
            device = 'cuda:0',
            # name = 'local_ct_5',
            plots = True,
            resume = True,
            exist_ok = True,

            # augmentation
            mosaic = 1,
            fliplr = 0.5,
            mixup = 0,
            hsv_v = 0.1,
            scale = 0.2, # mô phỏng vị trí gần - xa máy chụp
            degrees = 5,
            translate = 0.2,
            
            
            # contrastive
            use_contrastive = True,
            use_conf_aware = False
            )