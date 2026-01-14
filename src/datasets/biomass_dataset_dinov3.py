import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# original
import sys
sys.path.append(r"..")
from datasets.transforms_dinov3 import clean_image

class BiomassDataset(Dataset):
    def __init__(self, config, df, transform, img_dir):
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        self.paths = df['image_path'].values
        self.labels = df[config.all_target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        label = torch.from_numpy(self.labels[idx])
        return left, right, label