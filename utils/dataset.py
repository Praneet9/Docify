from os import listdir
from os.path import splitext
from glob import glob

from utils.image_aug import flip, add_gaussian_noise, add_uniform_noise, change_brightness, normalization2

import torch
from torch.utils.data import Dataset

import numpy as np
from random import randint
import cv2

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.height = 480
        self.width = 360

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, mask):
        
        # Augmentation
        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        img = flip(img, flip_num)
        mask = flip(mask, flip_num)
        
        # Noise Determine {0: Gaussian_noise, 1: uniform_noise
        if randint(0, 1):
            # Gaussian_noise
            gaus_sd, gaus_mean = randint(0, 20), 0
            img = add_gaussian_noise(img, gaus_mean, gaus_sd)
        else:
            # uniform_noise
            l_bound, u_bound = randint(-20, 0), randint(0, 20)
            img = add_uniform_noise(img, l_bound, u_bound)
        
        # Brightness
        pix_add = randint(-20, 20)
        img = change_brightness(img, pix_add)
        
        # Normalize the image
        img = normalization2(img, max=1, min=0)
        
        
        # Normalize mask to only 0 and 1
        mask = mask/255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
            
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        
        return img, mask

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = cv2.imread(mask_file[0])
        img = cv2.imread(img_file[0])
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]

        img, mask = self.preprocess(img, mask)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }