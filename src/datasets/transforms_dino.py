
import albumentations as A
import torch.optim as optim

import cv2
from albumentations.pytorch import ToTensorV2

def get_transforms(config, is_train):
    if is_train:
        return A.Compose([
            # 切り抜き後のサイズ, 切り抜きサイズの範囲
            A.RandomResizedCrop((config.img_h, config.img_w), scale=(0.8, 1.0)), # random crop
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(config.img_h, config.img_w),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    
def get_two_stream_transforms(config, is_train):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            A.Resize(224, 224),  # ensure DINOv2 input size
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ])