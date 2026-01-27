import os
from uuid import uuid4

from ultralytics import YOLO

# finetune step 1 - freeze
model_path = 'runs/benchmark/heatmap_sigmoid_yolo11m-hm-attn.yaml/weights/best.pt'
model = YOLO(model_path)
project_name = str(uuid4()) + 'step_1'
model.train(data='/media/ssd220/ty/xray/datasets/dataset.yaml',
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 16,
            cos_lr = True,
            multi_scale = False,
            workers = 4,
            freeze = [i for i in range(12, 24)],
            lr0 = 0.005,
            lrf = 0.005,

            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,
            hm = 10,
            ct = 10,

            # others
            # device = 'cuda:0',
            name = f'{project_name}',
            # name = f'dev',
            plots = True,
            resume = False,
            exist_ok = False,

            # augmentation
            mosaic = 0,
            fliplr = 0.5,
            mixup = 0,
            hsv_v = 0.1,
            # affine transforms
            scale = 0.2, 
            degrees = 10,
            translate = 0.2,
            flipud = 0.5,
            cutmix = 0,

            hm_scales = [8, 16, 32],
            contrastive = True
            )

# finetune step 2 - unfreeze
model_path = f'runs/heatmap/{project_name}/weights/best.pt'
model = YOLO(model_path)
project_name = project_name + 'step_2'
model.train(data='/media/ssd220/ty/xray/datasets/dataset.yaml',
            # hyp
            epochs = 500,
            imgsz = 640,
            batch = 16,
            cos_lr = True,
            multi_scale = False,
            workers = 4,
            lr0 = 0.005,
            lrf = 0.005,

            # loss
            box = 7.5,
            cls = 0.5,
            dfl = 1.5,
            hm = 10,
            ct = 10,

            # others
            # device = 'cuda:0',
            name = f'{project_name}',
            # name = f'dev',
            plots = True,
            resume = False,
            exist_ok = False,

            # augmentation
            mosaic = 0,
            fliplr = 0.5,
            mixup = 0,
            hsv_v = 0.1,
            # affine transforms
            scale = 0.2, 
            degrees = 10,
            translate = 0.2,
            flipud = 0.5,
            cutmix = 0,

            hm_scales = [8, 16, 32],
            contrastive = True
            )