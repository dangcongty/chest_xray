from ultralytics import YOLO

# model = YOLO('ultralytics/cfg/models/11/yolo11l-p2.yaml')
model = YOLO('runs/detect/single_cls4/weights/best.pt')
# model = YOLO('runs/detect/train23/weights/best.pt')
# model = YOLO('runs/detect/train56/weights/best.pt')
model.train(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 16,
            cos_lr = True,
            multi_scale = False,
            single_cls = True,


            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,
            # cls = 5,
            lr0 = 0.1,
            lrf = 0.1,

            # optimizer='Adam',

            # others
            device = 'cuda:1',
            name = 'single_cls',
            plots = True,
            resume = False,
            exist_ok = False,

            # augmentation
            mosaic = 1,
            fliplr = 0.5,
            mixup = 0,
            hsv_v = 0.1,
            # affine transforms
            scale = 0.2, # mô phỏng vị trí gần - xa máy chụp
            degrees = 10,
            translate = 0.2,
            flipud = 0.5,
            cutmix = 0.2
            
            # contrastive
            # use_contrastive = True,
            # use_conf_aware = False
            )