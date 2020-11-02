import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import (
    Compose, Normalize, ToTensor)
#sys.path.append("C:/Users/Tati/Documents/challenges/spacenet/progs/src")      
from .. configs import IMG_SIZE


def pad_x32(image, **kwargs):
    h, w = image.shape[:2]

    pad_h = np.ceil(h / 32) * 32 - h
    pad_w = np.ceil(w / 32) * 32 - w

    pad_h_top = int(np.floor(pad_h / 2))
    pad_h_bot = int(np.ceil(pad_h / 2))
    pad_w_top = int(np.floor(pad_w / 2))
    pad_w_bot = int(np.ceil(pad_w / 2))

    padding = ((pad_h_top, pad_h_bot), (pad_w_top, pad_w_bot), (0, 0))
    padding = padding[:2] if image.ndim == 2 else padding
    image = np.pad(image, padding, mode='constant', constant_values=0)

    return image


tensor_transform = Compose([
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])


d4_geometric = A.Compose([                        
			            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, 
                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                    	# D4 Group augmentations
                    	A.HorizontalFlip(p=0.5),
                    	A.VerticalFlip(p=0.5),
                    	A.RandomRotate90(p=0.5),
                    	A.Transpose(p=0.2),
                    	# crop and resize  
                    	A.RandomSizedCrop((IMG_SIZE-100, IMG_SIZE), IMG_SIZE, IMG_SIZE, w2h_ratio=1.0, 
                                        interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.2),                   	      	
                    	])


d4_tansforms = A.Compose([                       
                        # D4 Group augmentations
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Transpose(p=0.5),                        
			            ])

normalise = A.Normalize()


resize_norm = A.Compose([
                    A.SmallestMaxSize(IMG_SIZE, interpolation=0, p=1.),
                    A.RandomCrop(IMG_SIZE, IMG_SIZE, p=1.), 
                    A.Normalize(),
                    ])
        

geometric_transforms = A.Compose([                    
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, 
                       interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                    A.RandomSizedCrop((int(0.5*IMG_SIZE), IMG_SIZE), IMG_SIZE, IMG_SIZE),
                    # D4 Group augmentations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.2),                    
                    ])


train_light = A.Compose([                    
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),                    
                    ])
                    

train_medium = A.Compose([ 
            A.ShiftScaleRotate(shift_limit=0., scale_limit=0.2, rotate_limit=0, p = 0.5),          
            A.RandomSizedCrop((int(0.2*IMG_SIZE), int(0.5*IMG_SIZE)), IMG_SIZE, IMG_SIZE, p = 1),

            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.7),
                    A.GaussNoise(p=0.2),                 
                    A.RandomGamma(p=0.2),                    
                ],
                p=0.7),

            # D4 Group augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),                  
        ])


valid_flip = [A.HorizontalFlip()]


valid_ade = A.Compose([
            A.SmallestMaxSize(IMG_SIZE, p=1.),
            A.Lambda(name="Pad32", image=pad_x32, mask=pad_x32),   
            A.Normalize(),         
        ])


pad928 = A.Compose([
        A.PadIfNeeded(min_height=928, min_width=928)
        ])
# from bloodaxe 
# https://github.com/BloodAxe/Catalyst-Inria-Segmentation-Example/blob/master/inria/augmentations.py
crop_transform = A.Compose([A.RandomSizedCrop((int(0.5*IMG_SIZE), IMG_SIZE), IMG_SIZE, IMG_SIZE),                
            ])


safe_augmentations = A.Compose([A.HorizontalFlip(), A.RandomBrightnessContrast()])

light_augmentations = A.Compose([
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.OneOf([
                    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT),
                    A.IAAAffine(),
                    A.IAAPerspective(),
                    A.NoOp()
                ]),
                A.HueSaturationValue(),                
            ])


medium_augmentations = A.Compose([
                    A.HorizontalFlip(),
                    A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT),
                    # Add occasion blur/sharpening
                    A.OneOf([A.GaussianBlur(), A.IAASharpen(), A.NoOp()]),
                    # Spatial-preserving augmentations:
                    A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=5), A.NoOp()]),
                    A.GaussNoise(),
                    A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(), A.HueSaturationValue(), A.RGBShift(), A.RandomGamma()]),
                    # Weather effects
                    A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),                   
            ])


hard_augmentations = A.Compose([
                    A.RandomRotate90(),
                    A.Transpose(),
                    A.RandomGridShuffle(),
                    A.ShiftScaleRotate(
                        scale_limit=0.3, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0
                    ),
                    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, alpha_affine=5, mask_value=0, value=0),
                    # Add occasion blur
                    A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.IAAAdditiveGaussianNoise(), A.NoOp()]),
                    # D4 Augmentations
                    A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=10), A.NoOp()]),
                    # Spatial-preserving augmentations:
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(brightness_by_max=True),
                            A.CLAHE(),
                            A.HueSaturationValue(),
                            A.RGBShift(),
                            A.RandomGamma(),
                            A.NoOp(),
                        ]
                    ),
                    # Weather effects
                    A.OneOf([A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1), A.NoOp()]),                    
            ])           

# dictionary of transforms
TRANSFORMS = {
    "pad" : pad928,
    "d4": d4_tansforms,
    "normalise": normalise,
    "resize_norm": resize_norm,
    "geometric": geometric_transforms,
    "d4_geometric": d4_geometric,
    "light": train_light,
    "medium": train_medium,
    "hflip": valid_flip,
    "ade_valid": valid_ade,     
    "tensor_norm": tensor_transform,
    "flip_bright": safe_augmentations,
    "inria_light": light_augmentations,
    "inria_medium": medium_augmentations,
    "inria_hard": hard_augmentations,
    "inria_valid": safe_augmentations,
}
