from glob import glob

from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_p2_3k_bg/weights/best.pt')


# with open('datasets/process/val.txt', 'r') as f:
#     img_paths = f.readlines()

# for im_path in img_paths:
    
#     results = model(im_path, 
#                     imgsz = 1280,
#                     device = 'cuda:1',
#                     conf = 0.1,
#                     verbose = False)

#     results[0].save('test.jpg')

model.val(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
            imgsz = 640,
            batch=16)