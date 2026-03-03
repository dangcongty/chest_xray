import torch

from ultralytics import YOLO

model = YOLO("runs/heatmap/from_scratch/weights/best.pt")
# model.load("yolo11m_pretrained.pt")
# model.load("yolo11m.pt")

model.train(data='/mnt/workspace/ty/xray/datasets/dataset.yaml',
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 48,
            cos_lr = True,

            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,

            # others
            device = 'cuda:0',
            # project = 'runs',
            name = f'dev',
            plots = True,
            resume = False,
            exist_ok = False,

            # augmentation
            mosaic = 0,
            fliplr = 0.5,
            mixup = 0,
            # hsv_v = 0.1,
            # affine transforms
            scale = 0.1, 
            degrees = 2,
            translate = 0.05,
            # flipud = 0.5,
            cutmix = 0,

            # contrastive
            use_ct = False,

            # heatmap
            hm = 10
            )