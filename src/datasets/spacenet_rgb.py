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

from .transforms import TRANSFORMS
from .. configs import IMG_SIZE, TRAIN_JSON, TRAIN_META, TRAIN_RGB, TRAIN_MASKS

warnings.simplefilter("ignore")


class RGBDataset(Dataset):
    """

    SpaceNet 6 RGB Dataset

    Args:         
        images_dir: directory with RGB inputs
        masks_dir: directory with binary masks
        labels_df: true labels (as polygons)  
        img_size: the desired image size to resize to for prograssive learning
        transforms: the name of transforms setfrom the transfroms dictionary  
        debug: if True, runs debugging on a few images. Default: 'False'   
        normalise: if True, normalise images. Default: 'True'

    """
    def __init__(self, 
                images_dir: str,                 
                masks_dir: str,     
                labels_df: pd.DataFrame,           
                img_size: int = IMG_SIZE,                 
                transforms: str ='train', 
                normalise: bool = True,                        
                debug: bool = False,               
                ):
        super(RGBDataset, self).__init__()  # inherit it from torch Dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir       
        self.debug = debug
        self.normalise = normalise        
        self.img_size = img_size
        self.transforms = transforms
        self.ids = labels_df.ImageId.values        
        # select a subset for the debugging
        if self.debug:
            self.ids = self.ids[:160]
            print('Debug mode, samples: ', self.ids[:10])  

    def __len__(self):
        return len(self.ids)        
       
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        image_path = os.path.join(self.images_dir, "SN6_Train_AOI_11_Rotterdam_PS-RGB_{}.tif".format(sample_id))              
        # for preprocessed masks, 900x900 binary
        mask_path = os.path.join(self.masks_dir, 'SN6_Train_AOI_11_Rotterdam_Buildings_{}.npy'.format(sample_id))        
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)      
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         
            mask = np.load(mask_path)
            # resize if needed
            #image = cv2.resize(image, (self.img_size, self.img_size))
            #mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print(f'image.shape: {image.shape}')
            print(f'Missing Id: {sample_id}')
            image = np.zeros((self.img_size, self.img_size, 3), np.uint8)
            mask = np.zeros((self.img_size, self.img_size), np.uint8)    
            pass
       
        # augment
        if self.transforms is not None: 
            augmented = self.transforms(image=image, mask=mask)  
            image = augmented['image']
            mask = augmented['mask'] 
            
        # normalise
        if self.normalise:
            image = normalize(image)  

        # post-processing
        image = image.transpose(2,0,1).astype(np.float32) # channels first
        target = mask.astype(np.uint8)  # single channel, int 
        target = np.expand_dims(target, axis=0)
        #print(target.shape)
        
        image = torch.from_numpy(image) 
        target = torch.from_numpy(target)
 
        return image, target, sample_id
        

def normalize(img: np.array, mean: list=[0.485, 0.456, 0.406], std: list=[0.229, 0.224, 0.225], max_value: int=255) -> np.array:
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


def test_dataset() -> None:
    """Helper to vizualise a sample from the data set"""
    df = pd.read_csv(TRAIN_META)
    train_dataset = RGBDataset(
                images_dir = TRAIN_RGB,                 
                masks_dir = TRAIN_MASKS,
                labels_df = df, 
                img_size  = 512,                
                transforms= None,
                normalise = True,              
                debug     = True,   
    ) 
    im, target, sample_id = train_dataset[10]
    plot_img_target(im, target, sample_id, fig_num = 1)                


def plot_img_target(image: torch.Tensor, target: torch.Tensor, sample_token: str = '', fig_num: int = 1) -> None:
    """Helper to plot image and target together"""
    image = image.numpy()
    print(image.shape)
    # transpose the input volume CXY to XYC order
    image = image.transpose(1,2,0)     
    image = np.rint(image).astype(np.uint8)
    
    target = target.numpy()
    print(target.shape)
    target =np.rint(target*255).astype(np.uint8)               
    target_as_rgb = np.repeat(target[...,None], 3, 2) # repeat array for three channels
    print(target_as_rgb.shape)

    plt.figure(fig_num, figsize=(12,6))        
    plt.imshow(np.hstack((image, target_as_rgb))) 
    plt.title(sample_token)
    plt.show()


def test_dataset_augs(img_size: int=224, transforms: dict = TRANSFORMS["d4"]) -> None:
    """Helper to test data augmentations"""
    df = pd.read_csv(TRAIN_META)
    train_dataset = RGBDataset(
                images_dir = TRAIN_RGB,                 
                masks_dir = TRAIN_MASKS,
                labels_df = df, 
                img_size  = img_size,                 
                transforms= transforms,
                normalise = False,           
                debug     = True,  
    )
    for count in range(5):
        # get dataset sample and plot it
        im, target, sample_token = train_dataset[10]
        plot_img_target(im, target, sample_token, fig_num = count+1)


if __name__ == "__main__":
    test_dataset()
    test_dataset_augs(img_size=512, transforms = TRANSFORMS["medium"])