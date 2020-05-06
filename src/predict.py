import os
import scipy
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from utils.utils import set_seed, load_model 
from datasets.dataset_ade20k import ADE20KDataset
from models.get_models import get_unet, get_fpn
from configs import TRAIN_IMAGES, TRAIN_MASKS, NUM_CLASSES, IMG_SIZE, VALID_IMAGES, VALID_MASKS      


def plot_predictions(input_image, prediction, predictions_dir: str, checkpoint_name: str, n_images: int = 2, apply_softmax: bool = True):
    """
    Takes as input 3 PyTorch tensors, plots the input image, predictions and targets
    
    Args: 
        input_image: the input image
        predictions: the predictions thresholded at 0.5 probability        
    """
    # Only select the first n images
    prediction = prediction[:n_images]
    target = target[:n_images]
    input_image = input_image[:n_images]

    prediction = prediction.detach().cpu().numpy()
    if apply_softmax:
        prediction = scipy.special.softmax(prediction, axis=1)

    preds = np.hstack(prediction)
    preds_rgb = np.repeat(preds[..., None], 3, axis=2) # repeat for 3 channels to plot
    thresholded_pred = np.repeat(preds_rgb[..., None] > 0.5, 3, axis=2)
    input_im = np.hstack(input_image.cpu().numpy().transpose(0,2,3,1))     
    
    if input_im.shape[2] == 3:
        input_im_grayscale = np.repeat(input_im.mean(axis=2)[..., None], 3, axis=2)
        overlayed_im = (input_im_grayscale*0.6 + preds_rgb*0.7).clip(0,1)
    else:
        input_map = input_im[...,3:]
        overlayed_im = (input_map*0.6 + preds_rgb*0.7).clip(0,1)    

    fig = plt.figure(figsize=(12,26))
    plot_im = np.vstack([input_im[...,:3], overlayed_im, preds_rgb, target_rgb]).clip(0,1).astype(np.float32)
    plt.imshow(plot_im)
    plt.axis("off")
    plt.savefig(os.path.join(predictions_dir,"preds_{}_epoch_{}.png".format(model_name, epoch))) 


def run_predict(model, test_folder: str, checkpoint_name: str, predictions_dir: str, debug: bool = False, img_size: int = IMG_SIZE, 
                batch_size: int = 8, num_workers: int = 4):
    """
    Run predicitons of the model checkpoint
    
    Args: 
        model : PyTorch model
        test_folder: directory with test images        
        checkpoint_name: name of the checkpoint
        predictions_dir: directory for saving predictions
        debug: if True, runs the debugging on few images 
        img_size: size of images for training (for pregressive learning)
        batch_size: number of images in batch
        num_workers: number of workers available   
        plot_preds: boolean flag, if plot oof predictions and save them as png 
        save_oof: boolean flag, if calculate oof predictions and save them in pickle         
    Output:
        oof: test predicitons         
    """ 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)       
    print(f'\n {checkpoint_name} \n')
    
    # test dataset
    test_dataset = ADE20KTestDataset(
                images_dir = test_folder, 
                img_size = img_size, 
                transforms = None,           
                debug = debug,)                           

    dataloader_test = DataLoader(test_dataset,
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=False)
    print('{} test images'.format(len(test_dataset)))

    with torch.no_grad():
        model.eval()        
        progress_bar = tqdm(dataloader_test, total=len(dataloader_test))
        oof = collections.defaultdict(list)

        for iter_num, (img, sample_ids) in enumerate(progress_bar):
            img = img.to(device)  
            prediction = model(img)             
            
            prediction = F.softmax(prediction, dim=1)             
            # Visualise the first prediction
            if iter_num == 0 and plot_preds:
                plot_predictions(img, prediction, predictions_dir, checkpoint_name)    
            oof["preds"].append(prediction.cpu().numpy().copy())              
    if save_oof:  
        pickle.dump(oof, open(f"{predictions_dir}/{checkpoint_name}.pkl", "wb"))    
    return oof
    


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument    
    arg('--predictions-dir', type=str, default='../../../oof', help='Directory for saving oof preds')
    arg('--results-dir', type=str, default='../../../results/runs', help='Directory for saving checkpoints')
    arg('--test-dir', type=str, default='../../../data/test', help='Directory for test images')
    arg('--image-size', type=int, default=224, help='Image size for training')
    arg('--batch-size', type=int, default=4, help='Batch size during training')
    arg('--num-workers', type=int, default=4, help='Number of workers for dataloader. Default = 4.')
    arg('--epochs', type=int, default=10, help='Epoch to run')    
    arg('--checkpoint', type=str, default=None, help='Checkpoint filename with initial model weights')
    arg('--debug', type=bool, default=True)
    args = parser.parse_args() 
                
    set_seed(seed=1234)
    
    checkpoint_file = os.path.join(args.results_dir, args.checkpoint) 
    # load checkpoint
    checkpoint = load_model(checkpoint_file)
    checkpoint_epoch = checkpoint['epoch']
   
