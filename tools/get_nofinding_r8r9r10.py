import pandas as pd
from tqdm import tqdm

df = pd.read_csv('datasets/annotations_train.csv')

nofindings = []
for idx, row in tqdm(df.iterrows()):
    if row['rad_id'] in ['R8', 'R9', 'R10'] and row['class_name'] == 'No finding':
        nofindings.append(f'datasets/process/images/{row["image_id"]}.png\n')
nofindings = set(nofindings)

with open('datasets/process/nofinding_r8r9r10', 'w') as f:
    for path in nofindings:
        f.write(path)