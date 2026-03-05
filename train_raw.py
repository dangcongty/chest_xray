import torch

from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11m.yaml")
model.load("yolo11m.pt")
# model = YOLO("runs/heatmap/from_yolo11m_pretraine_1k_bg.pt2/weights/best.pt")

model.train(data='/mnt/workspace/ty/xray/datasets/dataset.yaml',
            epochs = 500,
            imgsz = 640,
            batch = 16,
            cos_lr = True,

            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,

            # others
            device = 'cuda:1',
            # project = 'runs',
            name = f'raw',
            plots = True,
            resume = False,
            exist_ok = False,

            # augmentation
            mosaic = 1,
            fliplr = 0.5,
            mixup = 0,
            # hsv_v = 0.1,
            # affine transforms
            scale = 0.1, 
            degrees = 2,
            translate = 0.05,
            flipud = 0.5,
            cutmix = 0,

            # contrastive
            use_ct = False,

            # heatmap
            hm = 0,
            use_hm = False
            )