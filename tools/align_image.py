import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

os.makedirs('datasets/align/images', exist_ok=True)
os.makedirs('datasets/align/labels', exist_ok=True)
translate_ratios = []
for mask_path in tqdm(glob('datasets/masks/*')):
    img_path = f'datasets/images/{os.path.basename(mask_path)[:-4]}.png'
    label_path = f'datasets/labels/{os.path.basename(mask_path)[:-4]}.txt'

    mask = np.load(mask_path)
    img = cv2.imread(img_path)
    with open(label_path, 'r') as f:
        label = f.readlines()


    size = 1280
    scale = size/mask.shape[1]
    px = np.argwhere(mask[0]<0.5)[:, ::-1]
    current_pos_x = px[:,0].mean()/mask.shape[1] # norm
    
    translate_ratio = 0.5 - current_pos_x

    translation_matrix = np.float32([
        [1, 0, int(translate_ratio*1280)],
        [0, 1, 0]
    ])
    translated_image = cv2.warpAffine(img, translation_matrix, (size, size))
    translate_ratios.append(int(translate_ratio*1280))

np.save('datasets/translate_ratios.npy', translate_ratios)