# from ultralytics import YOLO

# model = YOLO('runs/detect/train23/weights/best.pt')
# model.val(data = '/media/ssd220/ty/xray/datasets/dataset.yaml', 
#             imgsz = 640,
#             mosaic = 0,
#             batch=64,
#             device = 'cuda:1')

a = [
"Nguyễn Nguyễn Đăng Khoa",
"Huỳnh Quốc Thắng",
"Trương Đình Khoa",
"Đinh Xuân Huy",
"Trần Quốc Huy",
"Đồng Gia Sang",
"Trịnh Duy Bách",
"Bùi Quang Huy",
"Nguyễn Tiến Cường",
"Bùi Vân Anh",
"Huỳnh Lê Thanh Liêm",
"Lê Hữu Trực",
"Trần Vũ Khanh",
"Võ Hữu Lộc",
"Hoàng Đức Tuấn",
"Nguyễn VĂn Tiến ĐẠT",
"Nguyễn Trung Nguyên",
"Bùi Đình Khôi"]


import numpy as np

print(np.random.choice(a, size=10))
