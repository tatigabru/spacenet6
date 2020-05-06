import os
import sys
import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..configs import (IMG_SIZE, TRAIN_JSON, TRAIN_MASKS, TRAIN_META, TRAIN_RGB,
                     TRAIN_SAR, TRAIN_DIR)
from .transforms import TRANSFORMS
 
warnings.simplefilter("ignore")


class SARDataset(Dataset):
    """
    SpaceNet 6 SAR Dataset

    Args:         
        sars_dir: directory with SARs inputs
        masks_dir: directory with binary masks
        labels_df: true labels (as polygons)   
        img_size: the desired image size to resize to for prograssive learning
        transforms: the name of transforms setfrom the transfroms dictionary  
        debug: if True, runs debugging on a few images. Default: 'False'   
        normalise: if True, normalise images. Default: 'True'

    """
    def __init__(self, 
                sars_dir: str,                 
                masks_dir: str,     
                labels_df: pd.DataFrame,           
                img_size: int = 512,                 
                transforms: str ='valid', 
                preprocess: bool = True,
                normalise: bool = True,                         
                debug: bool = False,            
                ):

        super(SARDataset, self).__init__()  # inherit it from torch Dataset
        self.sars_dir = sars_dir
        self.masks_dir = masks_dir       
        self.debug = debug
        self.preprocess = preprocess
        self.normalise = normalise        
        self.img_size = img_size
        self.transforms = transforms
        self.ids = labels_df.ImageId.values    
        #sar_ids = os.listdir(sars_dir)
        #self.ids = [s[41:-4] for s in sar_ids]
        # select a subset for the debugging
        if self.debug:
            self.ids = self.ids[:160]
            print('Debug mode, samples: ', self.ids[:10])  

    def __len__(self):
        return len(self.ids)        
       
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        image_path = os.path.join(self.sars_dir, "SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{}.tif".format(sample_id))              
        # for preprocessed masks, 900x900 binary
        mask_path = os.path.join(self.masks_dir, 'SN6_Train_AOI_11_Rotterdam_Buildings_{}.npy'.format(sample_id))        
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)               
            mask = np.load(mask_path)
            # pre-process, resize if needed
            image = cv2.resize(image, (self.img_size, self.img_size))            
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        except:
            print(f'Missing Id: {sample_id}')
            print("Unexpected error:", sys.exc_info()[0])
            image = np.ones_like((self.img_size, self.img_size, 4), np.uint8)
            mask = np.zeros((self.img_size, self.img_size), np.uint8)
            pass 
          
        # preprocess to 0..1
        if self.preprocess:   
            image = prep(image) 
        # augment
        if self.transforms is not None: 
            augmented = self.transforms(image=image, mask=mask)  
            image = augmented['image']
            mask = augmented['mask']           
        #print(f'image.shape: {image.shape}') 

        # normalise 
        if self.normalise:
            image = normalize(image, max_value=1)    
        # post-processing
        image = image.transpose(2,0,1).astype(np.float32) # channels first
        target = mask.astype(np.uint8)  
        target = np.expand_dims(target, axis=0) # single channel first
        #print(f'target.shape: {target.shape}')
           
        image = torch.from_numpy(image) 
        target = torch.from_numpy(target)
        
        return image, target, sample_id


def normalize(img: np.array, mean: list=[0.485, 0.456, 0.406, 0.406], std: list=[0.229, 0.224, 0.225, 0.225], max_value: float=92.88) -> np.array:
    """
    Noramalize image data to 0-1 range,
    then applymenaand std as in ImageNet pretrain, or any other
    """    
    mean = np.array(mean, dtype=np.float32)
    mean *= max_value
    std = np.array(std, dtype=np.float32)
    std *= max_value

    img = img.astype(np.float32)
    img = img - mean    
    img = img / std

    return img


def prep(img: np.array) -> np.array:
    """
    Normalize image data to 0-1 range,
    then applymenaand std as in ImageNet pretrain, or any other
    """    
    im_min = 0 # np.percentile(img, 2)
    im_max = np.percentile(img, 98)
    im_range = (im_max - im_min)
    #print(f'percentile 2 {im_min}, percentile 98 {im_max}, im_range {im_range}')
    img = np.clip(img, im_min, im_max)
    # normalise to the percentile
    img = img.astype(np.float32)
    img = img / im_range

    return img


def preprocess(img: np.array, mean: list=[0.485, 0.456, 0.406, 0.406], std: list=[0.229, 0.224, 0.225, 0.225], max_value: float=92.88) -> np.array:
    """
    Normalize image data to 0-1 range,
    then applymenaand std as in ImageNet pretrain, or any other
    """    
    im_min = np.percentile(img, 2)
    im_max = np.percentile(img, 99)
    im_range = (im_max - im_min)
    #print(f'percentile 2 {im_min}, percentile 99 {im_max}, im_range {im_range}')
    img = np.clip(img, im_min, im_max)

    mean = np.array(mean, dtype=np.float32)
    mean *= im_range
    std = np.array(std, dtype=np.float32)
    std *= im_range

    img = img.astype(np.float32)
    img = img - mean    
    img = img / std

    return img


def test_dataset():
    """Helper to vizualise a sample from the data set"""
    df = pd.read_csv(f'{TRAIN_DIR}folds.csv')
    train_dataset = SARDataset(
                sars_dir = TRAIN_SAR, 
                masks_dir = TRAIN_MASKS,
                labels_df = df, 
                img_size  = 512,                
                transforms= None,
                preprocess= True,
                normalise = True,              
                debug     = True,   
    ) 
    image, target, sample_id = train_dataset[5]
    plot_img_target(image, target, sample_id, fig_num = 1)                


def pauli_transform(sar_intensity: np.array):
    """Transform SAR to RGB
    Source: 
    """
    
    sar_rgb = []
    return sar_rgb


def plot_img_target(image: torch.Tensor, target: torch.Tensor, sample_token = None, fig_num = 1):
    """Helper to plot image and target together"""
    image = image.numpy()
    #print(image.shape)    
    # transpose the input volume CXY to XYC order
    image = image.transpose(1,2,0)     
    channel_4 = image[..., 3] # channel 4
    print("min max per channel", np.min(channel_4), np.max(channel_4))
    image = image[..., :3] # channels 1-2-3
    image = np.rint(image*255).astype(np.uint8)
    gray = np.rint(channel_4*255).astype(np.uint8)
    gray_as_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    target = target.numpy()
    if target.ndim == 3:
        target = np.squeeze(target, axis=0)

    target =np.rint(target*255).astype(np.uint8)               
    target_as_rgb = np.repeat(target[...,None], 3, 2) # repeat array for three channels

    plt.figure(fig_num, figsize=(18,6))        
    plt.imshow(np.hstack((image, gray_as_rgb, target_as_rgb))) 
    plt.title(sample_token)
    plt.show()


def test_dataset_augs(img_size: int=224, transforms = TRANSFORMS["d4"]):
    """Helper to test data augmentations"""
    df = pd.read_csv(f'{TRAIN_DIR}folds.csv')
    train_dataset = SARDataset(
                sars_dir = TRAIN_SAR, 
                masks_dir = TRAIN_MASKS,
                labels_df = df, 
                img_size  = img_size,                
                transforms= transforms,
                preprocess= True,
                normalise = False,              
                debug     = True,   
    ) 
    for count in range(5):
        # get dataset sample and plot it
        im, target, sample_id = train_dataset[5]
        plot_img_target(im, target, sample_id, fig_num = count+1)


if __name__ == "__main__":

    test_dataset()
    test_dataset_augs(img_size=512, transforms = TRANSFORMS["d4"])
