import glob
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

sys.path.append(os.getcwd())
from lung_seg.src.models import PretrainedUNet, UNet


class LungSegmentation:
    def __init__(self):
        
        self.device = 'cuda:1'


        self.model = PretrainedUNet(
            in_channels=1,
            out_channels=2, 
            batch_norm=True, 
            upscale_mode="bilinear"
        )
        self.model.load_state_dict(torch.load('lung_seg/models/unet-6v.pt'))
        self.model.eval()
        self.model.to(self.device)

    def get_data(self, path):
        img = Image.open(path).convert("P")
        img = torchvision.transforms.functional.resize(img, [512, 512])
        img = torchvision.transforms.functional.to_tensor(img) - 0.5
        img = img.to(self.device)
        return img
    


    def __call__(self):
        for img_path in glob.glob('datasets/process/images/*.png'):
            img = self.get_data(img_path)

            with torch.no_grad():
                mask = self.model(img.unsqueeze(0))
                mask = torch.argmax(mask, dim=1)
                mask = mask[0].float().cpu().numpy()

            mask = cv2.resize(mask, (1280, 1280))
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)

            # vis 
            vis = cv2.imread(img_path)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_bgr[:, :, 0] = np.where(mask_bgr[:, :, 0] == 1, 75, mask_bgr[:, :, 0])
            vis = vis*0.8 + mask_bgr
            print()
    

if __name__ == '__main__':
    lungseg = LungSegmentation()
    lungseg()