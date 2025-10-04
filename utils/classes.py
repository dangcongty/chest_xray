CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]

CLASSES_VN = [
    "Phình động mạch chủ",
    "Xẹp phổi",
    "Vôi hóa",
    "Tim to",
    "Đông đặc phổi",
    "Bệnh phổi kẽ",
    "Thâm nhiễm",
    "Đục/Mờ phổi",
    "Nốt/ Khối u",
    "Tổn thương khác",
    "Tràn dịch màng phổi",
    "Dày màng phổi",
    "Tràn khí màng phổi",
    "Xơ phổi",
]

CLASS_COLORS_BGR = {
    "Aortic enlargement": (180, 119, 31),   # #1f77b4
    "Atelectasis": (14, 127, 255),          # #ff7f0e
    "Calcification": (44, 160, 44),         # #2ca02c
    "Cardiomegaly": (40, 39, 214),          # #d62728
    "Consolidation": (189, 103, 148),       # #9467bd
    "ILD": (75, 86, 140),                   # #8c564b
    "Infiltration": (194, 119, 227),        # #e377c2
    "Lung Opacity": (127, 127, 127),        # #7f7f7f
    "Nodule/Mass": (34, 189, 188),          # #bcbd22
    "Other lesion": (207, 190, 23),         # #17becf
    "Pleural effusion": (147, 20, 255),     # #ff1493
    "Pleural thickening": (209, 206, 0),    # #00ced1
    "Pneumothorax": (0, 215, 255),          # #ffd700
    "Pulmonary fibrosis": (34, 139, 34),    # #228b22
}

IGNORE_CLASS = [
    "Clavicle fracture",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    " Rib fracture"
]

if __name__ == '__main__':
    import cv2
    import numpy as np

    img = np.zeros((500, 500, 3), dtype=np.uint8)
    for k, class_name in enumerate(CLASS_COLORS_BGR):
        print(k)
        color = CLASS_COLORS_BGR[class_name]
        img = cv2.putText(img, f'{class_name}', [150, 20 + k * 35], cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    cv2.imwrite('datasets/class_colors.jpg', img)