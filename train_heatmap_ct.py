import os
from uuid import uuid4

from ultralytics import YOLO

# model_path = '/media/ssd220/ty/xray/runs/heatmap/heatmap_ct_ctcls4/weights/best.pt'
model_path = 'ultralytics/cfg/models/11/yolo11m_enhance-hm.yaml'
model = YOLO(model_path)

project_name = str(uuid4()) + '_contrastive'
with open('training_name.txt', 'a') as f:
    f.write(project_name + '\n')
model.train(data='/media/ssd220/ty/xray/datasets/dataset.yaml',
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 16,

            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,
            hm = 0.01,
            ct = 5,
            ct_cls = 20,

            # others
            device = 'cuda:1',
            # name = f'increase_yolo_heads',
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

            hm_scales = [8, 16, 32],
            contrastive = True
            )