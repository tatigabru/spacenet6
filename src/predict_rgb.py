"""
Generate model preditions for the test data, spacenet 6
"""
import argparse
import collections
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch import Tensor

from .configs import *
# current project imports
from .datasets.spacenet import SARDataset
from .datasets.spacenet_rgb import RGBADataset
from .datasets.transforms import TRANSFORMS
from .models.get_models import get_unet
from .utils.utils import load_model, load_model_optim, set_seed, write_event


def generate_validation_preds(model: nn.Module, predictions_dir: str, checkpoint: str, debug: bool, fold: int=0, save_oof: bool=True, img_size: int=IMG_SIZE, 
                        num_workers: int = 2, batch_size: int = 2, gpu_number: int = 0):
    """
    Load model weights from the checkpoint, 
    Makes validation predictions and saves oof as images
    
        Args: 
            predictions_dir: directory for saveing predictions
            checkpoints_dir: directory with weights 
            model_name     : string name from the models configs listed in models.py file
            fold           : evaluation fold number, 0-3
            debug          : if True, runs debugging on few images
            img_size       : size of images for validation
            batch_size     : number of images in batch
            num_workers    : number of workers available
                   
    """
    device = torch.device(f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu')    
    print("device: ", device)    
    os.makedirs(predictions_dir, exist_ok=True)

    df = pd.read_csv(f'{TRAIN_DIR}folds.csv')
    df_val = df[df.fold == fold]
    # dataset for validation    
    valid_dataset = RGBDataset(
                images_dir = TRAIN_RGB,                 
                masks_dir = TRAIN_MASKS,
                labels_df = df_val, 
                img_size  = img_size,                
                transforms= TRANSFORMS["hflip"],
                normalise = True,           
                debug     = debug, 
    )            
    # dataloaders for train and validation    
    dataloader_valid = DataLoader(valid_dataset,
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=True)
    print('{} validation images'.format(len(valid_dataset)))  

    # load model checkpoint           
    try:
        print("load", checkpoint)
        model = load_model(model, checkpoint)
    except FileNotFoundError:
        break
    model = model.to(device)
    model.eval()

    progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))        
    for batch_num, (img, target, tile_ids) in enumerate(progress_bar):  # iterate over batches
        img = img.to(device)
        target = target.float().to(device)
        output = model(img)          
        # save oof for some batches
        if batch_num and batch_num == 0:
            output = torch.sigmoid(output) 
            output = output.cpu().numpy().copy()
            for num, pred in enumerate(output, start=0):
                tile_name = tile_ids[num]
                if pred.ndim == 3:
                        pred = np.squeeze(pred, axis=0)
                prob_mask = np.rint(pred*255).astype(np.uint8) 
                prob_mask_rgb = np.repeat(prob_mask[...,None], 3, 2) # repeat array for three channels    
                if debug: print(f"{predictions_dir}/{tile_name}.png")
                cv2.imwrite(f"{predictions_dir}/{tile_name}.png", prob_mask_rgb)    


def generate_test_preds(model: nn.Module, debug: bool, predictions_dir: str, checkpoint: str, save_oof: bool=True, img_size: int=IMG_SIZE, num_workers: int=2, batch_size: int=8):
    """
    Make test predicitons
    
    Args: 
        model_name : string name from the models configs listed in models.py file
        fold: evaluation fold number, 0-3
        debug: if True, runs the debugging on few images 
        epochs: number of epochs to train
        checkpoints_dir: directory with weights 
        from_epoch, to_epoch: the first ad last epochs for predicitions generation  
    """
    # creates directories for predicitons
    predictions_dir = f"{predictions_dir}/test_oof/"
    os.makedirs(predictions_dir, exist_ok=True)
    print("\n", model_name, "\n")
    
    # test dataset
    test_dataset = TestSARDataset(
                sars_dir  = TEST_SAR,          
                img_size  = img_size,                
                transforms= TRANSFORMS["hflip"],
                preprocess= True,
                normalise = True,              
                debug     = debug,   
    )       

    # dataloaders for train and validation    
    dataloader_test = DataLoader(test_dataset,
                                 num_workers=num_workers,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False)
    print("{} test images".format(len(dataset_test)))

    # load model checkpoint   
    print("Load model: ", checkpoint)
    try:
        model = load_model(model, checkpoint)
    except FileNotFoundError:
        break

    progress_bar = tqdm(dataloader_test, total=len(dataloader_test))  
    with torch.no_grad():        
        for batch_num, (img, target, tile_ids) in enumerate(progress_bar):  # iterate over batches
            img = img.to(device)
            target = target.float().to(device)
            output = model(img)  
            output = torch.sigmoid(output) 
            output = output.cpu().numpy().copy()
            for num, pred in enumerate(output, start=0):
                tile_name = tile_ids[num]
                if pred.ndim == 3:
                    pred = np.squeeze(pred, axis=0)
                prob_mask = np.rint(pred*255).astype(np.uint8) 
                prob_mask_rgb = np.repeat(prob_mask[...,None], 3, 2) # repeat array for three channels    
                if debug: print(f"{predictions_dir}/{tile_name}.png")
                cv2.imwrite(f"{predictions_dir}/{tile_name}.png", prob_mask_rgb)  
         

def plot_preds(input_image: Tensor, prediction: Tensor, target: Tensor, predictions_dir: str, img_name: str, n_images: int = 2) -> None:
    """
    Takes as input PyTorch tensors, plots the input image, predictions and targets
    
    Args: 
        input_image    : the input image
        predictions    : the predictions thresholded at 0.5 probability      
        target         : Tensor of the targets
        predictions_dir: directory with oof predictions
        n_images       : number of images to tack on the plot
        save_fig       : if true, saves the figure. Default: True

    """
    # Select the first n images in a batch
    prediction = prediction[:n_images]    
    prediction = torch.sigmoid(prediction)
    prediction.round() # binarise prediction
    target = target[:n_images]
    input_image = input_image[:n_images]

    prediction = prediction.detach().cpu().numpy()    
    preds = np.hstack(prediction)
    preds_rgb = np.repeat(preds[..., None], 3, axis=2) # repeat for 3 channels to plot
    #thresholded_pred = np.repeat(preds_rgb[..., None] > 0.5, 3, axis=2)
    
    input_image = input_image.cpu().numpy().transpose(0,2,3,1)    
    #input_im = np.hstack(input_image[2])  # use Exy channel only for visualisation
    #input_rgb = np.repeat(input_im[..., None], 3, axis=2) # repeat channel 3 times to plot
    overlayed_im = (input_image*0.6 + preds_rgb*0.7).clip(0,1)   

    target = target.detach().cpu().numpy()
    target = np.hstack(target)
    target_rgb = np.repeat(target[..., None], 3, axis=2) # repeat for 3 channels to plot 

    plt.figure(figsize=(12,26))
    plot_im = np.vstack([input_image, overlayed_im, preds_rgb, target_rgb]).clip(0,1).astype(np.float32)
    plt.imshow(plot_im)
    plt.axis("off")
    plt.savefig(os.path.join(predictions_dir, f"oof_{img_name}.png")) 
    #plt.show()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model", type=str, default="se_resnext101_dr0.75_512", help="String model name from models dictionary")
    arg("--seed", type=int, default=1234, help="Random seed")
    arg("--fold", type=int, default=0, help="Validation fold")
    arg("--checkpoints_dir", type=str, default="../../checkpoints", help="Directory for loading model weights")
    arg("--from-epoch", type=int, default=1, help="Resume training from epoch")
    arg("--to-epochs", type=int, default=15, help="Number of epochs to run")
    arg("--debug", type=bool, default=False, help="If the debugging mode")
    args = parser.parse_args()

    set_seed(args.seed)
    
    weights = f"{args.checkpoints_dir}/{args.model}_fold_{args.fold}/"

    predict_test(
        model_name=args.model, fold=args.fold, debug=args.debug, checkpoints_dir=weights, save_oof=True, img_size=IMG_SIZE, from_epoch=0, to_epoch=10
    )


if __name__ == "__main__":
    main()