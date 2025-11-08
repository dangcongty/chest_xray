import json
import os
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
model = model.eval().to(device)

# Load grayscale CXR image (e.g., 512x512)
outputs = {}
views = {
    0: 'AP',
    1: 'PA',
    2: 'lateral'
}
os.makedirs('dataset/masks', exist_ok=True)
for path in tqdm(glob('datasets/images/*.png')):
    img = cv2.imread(path, 0)
    x = model.preprocess(img)  # Built-in preprocessing
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()  # Add batch/channel dims

    with torch.inference_mode():
        out = model(x.to(device))

    outputs[path] = [views[int(out['view'].argmax())], float(out['view'].max())]
    mask = out['mask'].cpu().numpy()[0]
    np.save(f'dataset/masks/{os.path.basename(path)[:-4]}.npy', mask)

with open('result_view.json', 'w') as f:
    json.dump(outputs, f)



