


from ultralytics import YOLO

model_path = 'runs/detect/yolo_train_test_1set_yolo11l.yaml2/weights/best.pt'
model = YOLO(model_path)
model.predict(source='datasets/process/images/0046f681f078851293c4e710c4466058.png', visualize = True)