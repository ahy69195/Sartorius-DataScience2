import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2


class CellDataset(Dataset):
    def __init__(self, path, train=False, transform=None):
        # load path variables
        self.BASE_PATH = path
        self.mask_paths = os.listdir(f'{self.BASE_PATH}/masks_compiled')
        # Check if train or test and split data
        self.train = train
        self.mask_paths_train, self.mask_paths_test = train_test_split(self.mask_paths, test_size=0.2, random_state=42)
        if self.train:
            self.mask_paths = self.mask_paths_train
        else:
            self.mask_paths = self.mask_paths_test
        # Register any transforms
        self.transforms = transform

    def __len__(self):
        # num unique images
        return len(self.mask_paths)

    def __getitem__(self, index):
        # Load Mask
        mask = cv2.imread(f'{self.BASE_PATH}/masks_compiled/{self.mask_paths[index]}').astype(np.float32)
        # U-Net doesn't understand significance of 255 so just set it to 1.0
        mask[mask == 255.0] = 1.0
        # Load Image
        img = cv2.imread(f'{self.BASE_PATH}/train/{str(self.mask_paths[index])[0:12]}.png')
        # Apply any transforms to Image and Mask; return them with img_id
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            img = img.astype(np.float32)
            mask = mask.astype(np.float32)
        return self.mask_paths[index][0:12], img, mask
