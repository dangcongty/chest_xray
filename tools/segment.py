import os
import sys
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())
from lung_seg.src.data import Crop, LungDataset, Pad, Resize
from lung_seg.src.models import PretrainedUNet, UNet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
val_test_transforms = torchvision.transforms.Compose([
    Resize((512, 512)),
])

model = PretrainedUNet(
    in_channels=1,
    out_channels=2, 
    batch_norm=True, 
    upscale_mode="bilinear"
)
model.load_state_dict(torch.load('lung_seg/models/unet-6v.pt'))
model.eval()
model.to(device)

os.makedirs('datasets/process/segments', exist_ok=True)
os.makedirs('datasets/process/segments_vis', exist_ok=True)
for path in tqdm(glob('datasets/process/images/*.png')):
    img = Image.open(path)
    img = torchvision.transforms.functional.to_tensor(img) - 0.5
    img = img.to(device)[0, ...].unsqueeze(0).unsqueeze(0)

    outs = model(img)
    segs = torch.argmax(outs, dim=1)
    mask = segs[0].cpu().numpy()*125
    np.save(f'datasets/process/segments/{os.path.basename(path)}', mask)

    img_orig = img.squeeze().cpu().numpy()*255
    blend = mask + img_orig
    cv2.imwrite(f'datasets/process/segments_vis/{os.path.basename(path).replace("png", "jpg")}', blend)