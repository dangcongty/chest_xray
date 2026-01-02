

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

from ultralytics import YOLO

model_path = 'runs/heatmap/contrastive7/weights/best.pt'
model = YOLO(model_path)

with open('datasets/process/val_3k_bg.txt', 'r') as f:
    paths = f.readlines()

embs = []
gts = []
for path in tqdm(paths):
    path = path.strip()
    with open(path.replace('images', 'labels').replace('png', 'txt'), 'r') as f:
        lb = f.readlines()

    emb = model.predict(source=path, embed = [6])
    emb = emb[0].cpu().numpy()
    gts.append(1 if len(lb) else 0)
    embs.append(emb)

X = np.vstack([e.squeeze() for e in embs])  # shape: (N, D)
y = np.array(gts)                           # shape: (N,)

tsne = TSNE(
    n_components=2,
    perplexity=30,      # try 10–50 depending on N
    learning_rate=200,
    random_state=42
)

X_2d = tsne.fit_transform(X)


plt.figure(figsize=(8, 8))
plt.scatter(
    X_2d[y == 0, 0], X_2d[y == 0, 1],
    s=10, alpha=0.6, label='Background'
)
plt.scatter(
    X_2d[y == 1, 0], X_2d[y == 1, 1],
    s=10, alpha=0.6, label='Object'
)

plt.legend()
plt.title('t-SNE of YOLO Embeddings (Layer 6)')
plt.axis('off')
plt.savefig('embs_6.jpg')