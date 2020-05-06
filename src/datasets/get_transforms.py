import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import label2rgb
from torchvision.transforms import (
    Compose, Normalize, ToTensor)


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


class TransfromsCfgs():
    def __init__(
        self,
        img_size: tuple = (512, 512),   
        augs_name: str = "valid"      
        shift_limit: float=0.0625, 
        scale_limit: float=0.3, 
        rotate_limit: int=7,   
        crop_size: tuple = (512, 512),    
        hflip: bool = True,
        vflip:bool = False,  
        p: float = 0.5,
        normalise: bool = True,
    ):
        super(TransfromsCfgs, self).__init__()  # inherit it from torch Dataset
        self.img_size = img_size
        self.augs_name = augs_name
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.crop_size = crop_size
        self.p = p
        self.normalise = normalise
        

    def get_transforms(self):
        normalise = A.Normalize() 

        d4_tansforms = A.Compose([
                        A.SmallestMaxSize(self.img_size, interpolation=0, p=1.),
                        # D4 Group augmentations
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Transpose(p=0.5),        
                        A.Normalize(),
			            ])

        tensor_transform = Compose([
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

        d4_geometric = A.Compose([
                                A.SmallestMaxSize(self.img_size, interpolation=0, p=1.),
                                A.ShiftScaleRotate(shift_limit=self.shift_limit, self.scale_limit, rotate_limit=self.rotate_limit, 
                                interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                                # D4 Group augmentations
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.RandomRotate90(p=0.5),
                                A.Transpose(p=0.2),
                                # crop and resize  
                                A.RandomSizedCrop((self.crop_size[0], min(self.crop_size[1], self.img_size[0], self.img_size[1])), 
                                                    self.img_size[0], self.img_size[1], w2h_ratio=1.0, 
                                                    interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.2),                 
                                A.Normalize(),
                                ])

        resize_norm = A.Compose([
                    A.SmallestMaxSize(IMG_SIZE, interpolation=0, p=1.),
                    A.RandomCrop(self.img_size[0], self.img_size[1], p=1.), 
                    A.Normalize(),
                    ])        

        geometric_transforms = A.Compose([
                            A.SmallestMaxSize(self.img_size, interpolation=0, p=1.),
                            A.ShiftScaleRotate(shift_limit=self.shift_limit, self.scale_limit, rotate_limit=self.rotate_limit, 
                            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                            A.RandomSizedCrop((self.crop_size[0], min(self.crop_size[1], self.img_size[0], self.img_size[1])), self.img_size[0], self.img_size[1]),
                            # D4 Group augmentations
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.Transpose(p=0.2),                    
                            A.Normalize(),
                            ])

        train_light = A.Compose([
                            A.SmallestMaxSize(self.img_size, interpolation=0, p=1.),
                            A.RandomCrop(self.img_size[0], self.img_size[1], p=1.),
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.Normalize(),
                            ])

        train_light_show = A.Compose([
                            A.SmallestMaxSize(self.img_size, interpolation=0, p=1.),
                            A.RandomCrop(self.img_size[0], self.img_size[1], p=1.),
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(p=0.5),                    
                            ])

        train_medium = A.Compose([
                    A.SmallestMaxSize(self.img_size, interpolation=0, p=1.),
                    A.RandomCrop(self.img_size[0], self.img_size[1], p=1.),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=self.shift_limit, self.scale_limit, rotate_limit=self.rotate_limit, p=0.5),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=0.7),
                            A.Equalize(p=0.3),
                            A.HueSaturationValue(p=0.5),
                            A.RGBShift(p=0.5),
                            A.RandomGamma(p=0.4),
                            A.ChannelShuffle(p=0.05),
                        ],
                        p=0.9),
                    A.OneOf([
                        A.GaussNoise(p=0.5),
                        A.ISONoise(p=0.5),
                        A.MultiplicativeNoise(0.5),
                    ], p=0.2),  
                    A.Normalize(),         
                ])

        valid_ade = A.Compose([
                    A.SmallestMaxSize(self.img_size, p=1.),
                    A.Lambda(name="Pad32", image=pad_x32, mask=pad_x32),   
                    A.Normalize(),         
                ])       

        # from bloodaxe 
        # https://github.com/BloodAxe/Catalyst-Inria-Segmentation-Example/blob/master/inria/augmentations.py
        crop_transform = A.Compose([ #(image_size: Tuple[int, int], min_scale=0.75, max_scale=1.25, input_size=5000):
                A.OneOrOther(
                A.RandomSizedCrop((self.crop_size[0], min(self.crop_size[1], self.img_size[0], self.img_size[1])), 
                                   self.img_size[0], self.img_size[1]), 
                A.CropNonEmptyMaskIfExists(self.img_size[0], self.img_size[1]),
               ) 
            ])

        safe_augmentations = A.Compose([A.HorizontalFlip(), A.RandomBrightnessContrast(), A.Normalize()])

        light_augmentations = A.Compose([
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.OneOf([
                    A.ShiftScaleRotate(scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, border_mode=cv2.BORDER_CONSTANT),
                    A.IAAAffine(),
                    A.IAAPerspective(),
                    A.NoOp()
                ]),
                A.HueSaturationValue(),
                A.Normalize()
            ])

        medium_augmentations = A.Compose([
                    A.HorizontalFlip(),
                    A.ShiftScaleRotate(scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, border_mode=cv2.BORDER_CONSTANT),
                    # Add occasion blur/sharpening
                    A.OneOf([A.GaussianBlur(), A.IAASharpen(), A.NoOp()]),
                    # Spatial-preserving augmentations:
                    A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=5), A.NoOp()]),
                    A.GaussNoise(),
                    A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(), A.HueSaturationValue(), A.RGBShift(), A.RandomGamma()]),
                    # Weather effects
                    A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
                    A.Normalize(),
            ])

        hard_augmentations = A.Compose([
                    A.RandomRotate90(),
                    A.Transpose(),
                    A.RandomGridShuffle(),
                    A.ShiftScaleRotate(
                        scale_limit=self.scale_limit, rotate_limit=self.rotate_limit, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0
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
                    A.Normalize(),
            ])   

        TRANSFORMS = {
                "d4": d4_tansforms,
                "normalise": normalise,
                "resize_norm": resize_norm,
                "geometric": geometric_transforms,
                "d4_geometric": d4_geometric,
                "ade_light": train_light,
                "ade_medium": train_medium,
                "ade_valid": valid_ade,    
                "ade_show": train_light_show,
                "resize_norm": resize_norm,
                "flip_bright": safe_augmentations,
                "inria_light": light_augmentations,
                "inria_medium": medium_augmentations,
                "inria_hard": hard_augmentations,
                "inria_valid": safe_augmentations,
                }                   

        return TRANSFORMS[self.augs_name]