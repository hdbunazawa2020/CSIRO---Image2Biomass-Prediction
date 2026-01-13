import os
import cv2
import torch
from torch.utils.data import Dataset

class BiomassTwoStreamDataset(Dataset):
    """
    Dataset that splits each image into two square crops (left & right)
    Returns:
        - x_left: (3, 224, 224)
        - x_right: (3, 224, 224)
        - y: (3,) targets if training/validation, else None
    """
    def __init__(self, config, df, transforms=None, is_test=False, input_res=224, targets=["Dry_Green_g", "Dry_Total_g", "GDM_g"]):
        self.config = config
        self.df = df.reset_index(drop=True)
        self.image_paths = df["image_path"].values
        self.transforms  = transforms
        self.is_test   = is_test
        self.input_res = input_res

        if not is_test:
            self.targets_3 = df[targets].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def to_tensor(self, img):
        """
        Convert image to tensor and normalize to ImageNet
        """
        img = torch.from_numpy(img).permute(2, 0, 1).float() #/ 255.0 # (C, H, W) -> (W, H, C)
        return img

    def split_into_two_squares(self, img):
        """
        Split rectangular image into two square crops (left & right)
        """
        H, W, _ = img.shape
        if W < H:
            raise ValueError(f"Expected W >= H, got image shape {img.shape}")
        
        left  = img[:, :H, :]
        right = img[:, W-H:, :]
        return left, right

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        full_path  = os.path.join(self.config.input_dir, image_path)
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split into two squares
        x_left, x_right = self.split_into_two_squares(img)

        # Apply transforms if provided
        if self.transforms:
            x_left  = self.transforms(image=x_left)["image"]
            x_right = self.transforms(image=x_right)["image"]
        else:
            x_left  = self.to_tensor(x_left)
            x_right = self.to_tensor(x_right)

        if self.is_test:
            return x_left, x_right
        else:
            y = torch.tensor(self.targets_3[idx], dtype=torch.float32)
            return x_left, x_right, y