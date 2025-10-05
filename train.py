from ultralytics import YOLO

# model = YOLO('ultralytics/cfg/models/11/yolo11-p2.yaml')
model = YOLO('runs/detect/yolo11_p2/weights/best.pt')
model.train(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 16,
            cos_lr = True,

            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,


            # others
            device = 'cuda:0',
            name = 'yolo11_p2_3k_bg_rois',
            plots = True,
            resume = False,
            exist_ok = True,

            # augmentation
            mosaic = 0.0,
            fliplr = 0.0,
            mixup = 0,
            hsv_v = 0.1,
            scale = 0.0, # mô phỏng vị trí gần - xa máy chụp
            degrees = 0,
            translate = 0.0,
            
            
            # contrastive
            use_contrastive = True,
            use_conf_aware = False
            )