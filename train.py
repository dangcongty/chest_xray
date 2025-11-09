from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/11/yolo11n-hm.yaml')
# model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
model.train(data='/media/ssd220/ty/xray/datasets/dataset.yaml',
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 16,
            cos_lr = True,
            multi_scale = False,
            # single_cls = True,

            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,
            hm = 5,
            # cls = 5,
            lr0 = 0.1,
            lrf = 0.1,

            # optimizer='Adam',

            # others
            device = 'cuda:0',
            name = 'heatmap_09102025_',
            plots = True,
            resume = False,
            exist_ok = False,

            # augmentation
            mosaic = 1,
            fliplr = 0.5,
            mixup = 0,
            hsv_v = 0.1,
            # affine transforms
            scale = 0.2, 
            degrees = 10,
            translate = 0.2,
            flipud = 0.5,
            cutmix = 0.2

            
            
            )