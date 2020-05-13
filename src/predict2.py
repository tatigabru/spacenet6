"""
Generate model preditions for the test data
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
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torch import Tensor

from .configs import *
# current project imports
from .datasets.spacenet import SARDataset
from .datasets.transforms import TRANSFORMS
from .losses.bce_jaccard import BCEJaccardLoss
from .losses.dice import DiceLoss
from .losses.jaccard import JaccardLoss
from .models.get_models import get_fpn, get_unet
from .utils.f1_metric import binary_iou, buildings_f1_fast
from .utils.iou import binary_iou_numpy, binary_iou_pytorch, official_iou
from .utils.logger import Logger
from .utils.radam import RAdam
from .utils.utils import load_model, load_model_optim, set_seed, write_event



def generate_pedictions(predictions_dir: str, checkpoints_dir: str, debug: bool, fold: int=0, save_oof: bool=True, img_size: int=IMG_SIZE, 
                        num_workers: int = 2, batch_size: int = 2, from_epoch: int=0, to_epoch: int=10):
    """
    Loads model weights from the checkpoint, 
    Makes validation predictions and saves as images
    
        Args: 
            predictions_dir: directory for saveing predictions
            checkpoints_dir: directory with weights 
            model_name     : string name from the models configs listed in models.py file
            fold           : evaluation fold number, 0-3
            debug          : if True, runs debugging on few images
            img_size       : size of images for validation
            batch_size     : number of images in batch
            num_workers    : number of workers available
            from_epoch     : the first epoch for predicitions generation 
            to_epoch       : the last epoch for predicitions generation            
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)    
    os.makedirs(predictions_dir, exist_ok=True)

    df = pd.read_csv(f'{TRAIN_DIR}folds.csv')
    df_val = df[df.fold == fold]

     # dataset for validation
    valid_dataset = SARDataset(
                sars_dir  = TRAIN_SAR, 
                masks_dir = TRAIN_MASKS,
                labels_df = df_val, 
                img_size  = img_size,                
                transforms= TRANSFORMS["hflip"],
                preprocess= True,
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

    # load checkpoints
    for epoch_num in range(from_epoch, to_epoch):
        prediction_fn = f"{predictions_dir}/{epoch_num:03}.pkl"
        if os.path.exists(prediction_fn):
            continue
        print("epoch", epoch_num)
        # load model checkpoint
        checkpoint = (f"{checkpoints_dir}/{model_name}_{epoch_num:03}.pt")
        print("load", checkpoint)
        try:
            model = torch.load(checkpoint)
        except FileNotFoundError:
            break
        model = model.to(device)
        model.eval()

        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))        
        for batch_num, (img, target, tile_ids) in enumerate(progress_bar):  # iterate over batches
            img = img.to(device)
            target = target.float().to(device)
            output = model(img)  
            output = torch.sigmoid(output)   

            output = output.cpu().numpy().copy()

            for num, pred in enumerate(output, start=0):
                tile_name = tile_ids[num]
                prob_mask = (pred * 255).astype(np.uint8)
                if debug: print(f"{predictions_dir}/{tile_name}.png")
                cv2.imwrite(f"{predictions_dir}/{tile_name}.png", prob_mask)    


def predict_test(model_name: str, fold: int, debug: bool, checkpoints_dir: str, save_oof: bool=True, img_size: int=IMG_SIZE, from_epoch: int=0, to_epoch: int=10):
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
    predictions_dir = f"{RESULTS_DIR}/test_oof/{model_name}_fold_{fold}"
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
                                  drop_last=True)
    print("{} test images".format(len(dataset_test)))

    for epoch_num in range(from_epoch, to_epoch):
        # load model checkpoint
        checkpoint = f"{checkpoints_dir}/{model_name}_{epoch_num:03}.pt"
        print("load", checkpoint)
        try:
            retinanet = load_model(checkpoint)
        except FileNotFoundError:
            break

        data_iter = tqdm(enumerate(dataloader_test), total=len(dataloader_test))
        oof = collections.defaultdict(list)

        for  in data_iter:
            res = retinanet([data["img"].cuda().float(), data["annot"].cuda().float(), data["category"].cuda()], return_loss=False, return_boxes=True)
            nms_scores, global_class, transformed_anchors = res
            if save_oof:
                # predictions
                oof["gt_boxes"].append(data["annot"].cpu().numpy().copy())
                oof["gt_category"].append(data["category"].cpu().numpy().copy())
                oof["boxes"].append(transformed_anchors.cpu().numpy().copy())
                oof["scores"].append(nms_scores.cpu().numpy().copy())
                oof["category"].append(global_class.cpu().numpy().copy())

        if save_oof:  # save predictions
            pickle.dump(oof, open(f"{predictions_dir}/{epoch_num:03}.pkl", "wb"))


def plot_predictions(input_image: Tensor, prediction: Tensor, target: Tensor, predictions_dir: str, n_images: int = 2, save_fig: bool = True) -> None:
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
    # Only select the first n images in a batch
    prediction = prediction[:n_images]    
    # binarise prediction
    prediction = torch.sigmoid(prediction)
    prediction.round()
    target = target[:n_images]
    input_image = input_image[:n_images]

    prediction = prediction.detach().cpu().numpy()
    preds = np.hstack(prediction)
    preds_rgb = np.repeat(preds[..., None], 3, axis=2) # repeat for 3 channels to plot
    #thresholded_pred = np.repeat(preds_rgb[..., None] > 0.5, 3, axis=2)
    
    input_image = input_image.cpu().numpy().transpose(0,2,3,1)
    input_im = np.hstack(input_image[2])  # use Exy channel only for visualisation
    input_rgb = np.repeat(input_im[..., None], 3, axis=2) # repeat channel 3 times to plot
    overlayed_im = (input_rgb*0.6 + preds_rgb*0.7).clip(0,1)   

    target = target.detach().cpu().numpy()
    target = np.hstack(target)
    target_rgb = np.repeat(target[..., None], 3, axis=2) # repeat for 3 channels to plot 

    fig = plt.figure(figsize=(12,26))
    plot_im = np.vstack([input_rgb, overlayed_im, preds_rgb, target_rgb]).clip(0,1).astype(np.float32)
    plt.imshow(plot_im)
    plt.axis("off")
    if save_fig:
        plt.savefig(os.path.join(predictions_dir, f"preds_{checkpoint_name}.png")) 
    plt.show()

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