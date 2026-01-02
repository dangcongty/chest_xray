from glob import glob
import cv2
from tqdm import tqdm

for path in tqdm(glob('datasets/process/images/*')):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 640), cv2.INTER_CUBIC)
    cv2.imwrite(path, img)