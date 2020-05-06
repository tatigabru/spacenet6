import argparse
import collections
import os
import pickle

import numpy as np
#import sys
#sys.path.append("C:/Users/Tati/Documents/challenges/spacenet/progs/src")
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from sklearn.preprocessing import OneHotEncoder

from .configs import *
# current project imports
from .datasets.spacenet_rgb import RGBADataset
from .datasets.transforms import TRANSFORMS
from .models.get_models import get_fpn, get_unet
from .utils.iou import official_iou, binary_iou_pytorch
from .utils.f1_metric import buildings_f1_fast
from .utils.logger import Logger
from .utils.radam import RAdam
from .utils.utils import load_model, load_model_optim, set_seed


def train_runner(model: nn.Module, model_name: str, results_dir: str, debug: bool = False, img_size: int = IMG_SIZE,
                 learning_rate: float = 1e-2, fold: int = 0, 
                 epochs: int = 15, batch_size: int = 8, num_workers: int = 4, from_epoch: int = 0,
                 save_oof: bool = False):
    """
    Model training runner
    Args: 
        model : PyTorch model
        model_name : string name for model for checkpoints saving
        results_dir: directory to save results
        debug: if True, runs the debugging on few images 
        img_size: size of images for training (for pregressive learning)
        learning_rate: initial learning rate (default = 1e-2) 
        fold: training fold (default = 0)
        epochs: number of epochs to train
        batch_size: number of images in batch
        num_workers: number of workers available
        from_epoch: number of epoch to continue training    
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # creates directories for checkpoints, tensorboard and predicitons
    checkpoints_dir = f'{results_dir}/checkpoints/{model_name}'
    predictions_dir = f'{results_dir}/oof/{model_name}'
    tensorboard_dir = f'{results_dir}/tensorboard/{model_name}'
    validations_dir = f'{results_dir}/oof/{model_name}/val'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(validations_dir, exist_ok=True)
    logger = Logger(tensorboard_dir)
    print('\n', model_name, '\n')
    model = model.to(device)

    # datasets for train and validation
    df = pd.read_csv(f'{TRAIN_DIR}folds.csv')
    df_train = df[df.fold != fold]
    df_val = df[df.fold == fold]
    print(len(df_train.ImageId.values), len(df_val.ImageId.values))

    train_dataset = RGBADataset(
                images_dir = TRAIN_RGB,                 
                masks_dir = TRAIN_MASKS,
                labels_df = df_train, 
                img_size  = img_size,                 
                transforms= TRANSFORMS["medium"],
                normalise = True,           
                debug     = debug,  
    )    
    valid_dataset = RGBADataset(
                images_dir = TRAIN_RGB,                 
                masks_dir = TRAIN_MASKS,
                labels_df = df_val, 
                img_size  = img_size,                
                transforms= TRANSFORMS["d4"],
                normalise = True,           
                debug     = debug, 
    )            

    # dataloaders for train and validation
    dataloader_train = DataLoader(train_dataset,
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=True)                               

    dataloader_valid = DataLoader(valid_dataset,
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=True)
    print('{} training images, {} validation images'.format(len(train_dataset), len(valid_dataset)))

    # optimizers and schedulers
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = RAdam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.2)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.2)
    
    # training cycle
    print("Start training")
    all_losses = []
    for epoch in range(from_epoch, epochs + 1):
        print("Epoch", epoch)
        epoch_losses = []
        progress_bar = tqdm(dataloader_train, total=len(dataloader_train))

        # with torch.set_grad_enabled(True): --> sometimes people write it, idk
        for img, target, _ in progress_bar:
            img = img.to(device)
            target = target.to(device)            
            prediction = model(img)
           
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())

        # log loss history
        print("Epoch {}, Train Loss: {}".format(epoch, np.mean(epoch_losses)))
        all_losses.append(np.mean(epoch_losses))
        logger.scalar_summary('loss_train', np.mean(epoch_losses), epoch)

        # validate model after every epoch
        valid_metrics = validate(model, model_name, dataloader_valid, epoch,
                                 validations_dir, save_oof=save_oof)
        # logging metrics to tensorboard        
        logger.scalar_summary('loss_valid', valid_metrics['val_loss'], epoch)
        logger.scalar_summary('miou_valid', valid_metrics['miou'], epoch)
        
        # print current learning rate
        for param_group in optimizer.param_groups:
            print('learning_rate:', param_group['lr'])
        scheduler.step()

        # save model, optimizer and losses after every epoch
        checkpoint_filename = "{}_epoch_{}.pth".format(model_name, epoch)
        checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_filename)
        torch.save(
             {
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'loss': np.mean(epoch_losses),
                 'valid_loss': valid_metrics['val_loss'],
                 'valid_miou': valid_metrics['miou'],
             },
             checkpoint_filepath
        )


def validate_loss(model: nn.Module, model_name: str, dataloader_valid: DataLoader, epoch: int,
                  predictions_dir: str) -> float:
    """
    Validate model at the epoch end 
       
    Args: 
        model: current model 
        dataloader_valid: dataloader for the validation fold
        device: CUDA or CPU
        epoch: current epoch
        save_oof: boolean flag, if calculate oof predictions and save them in pickle 
        save_oof_numpy: boolean flag, if save oof predictions in numpy 
        predictions_dir: directory for saving predictions
    Output:
        loss_valid: total validation loss, history 
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model.eval()
        val_losses = []
        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))

        for img, target, _ in progress_bar:
            img = img.to(device)
            target = target.to(device)
            prediction = model(img)
            loss = F.cross_entropy(prediction, target)
            val_losses.append(loss.detach().cpu().numpy())
    print("Epoch {}, Valid Loss: {}".format(epoch, np.mean(val_losses)))

    return np.mean(val_losses)


def validate(model: nn.Module, model_name: str, dataloader_valid: DataLoader, epoch: int,
             predictions_dir: str, save_oof: bool = False):
    """
    Validate model at the epoch end 
       
    Args: 
        model: current model 
        dataloader_valid: dataloader for the validation fold
        device: GPU or CPU
        epoch: current epoch
        save_oof: boolean flag, if calculate oof predictions and save them in pickle 
        predictions_dir: directory for saving predictions
    Output:
         metrics: dictionary with validation metrics 
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model.eval()
        areas_intersection, areas_union = [], []
        val_losses = []
        progress_bar = tqdm(dataloader_valid, total=len(dataloader_valid))
        
        for _, (img, target, _) in enumerate(progress_bar):  # iterate over batches
            img = img.to(device)
            target = target.to(device)
            prediction = model(img)
            #loss = BCEWithLogitsLoss(prediction, target)
            loss = F.cross_entropy(prediction, target)
            val_losses.append(loss.detach().cpu().numpy())
            # get metrics
            prediction = F.softmax(prediction, dim=1)
            prediction = torch.argmax(prediction, dim=1)
            (area_intersection, area_union) = binary_iou_pytorch(prediction, target)
            areas_intersection.append(area_intersection.detach().cpu().numpy())
            areas_union.append(area_union.detach().cpu().numpy()) 
           
           # (area_intersection, area_union) = official_iou(prediction, target, numClass = 2)  # official ADE iou metric
           # print(f'area_intersection{area_intersection}, area_union {area_union}')
           # areas_intersection.append(area_intersection)
           # areas_union.append(area_union)  
         
            # save predictions to pictures        
            if save_oof:
                predictions = prediction.cpu().numpy().copy()
                for num, pred in enumerate(predictions, start=0):
                    tile_name = tile_ids[num]
                    prob_mask = (pred * 255).astype(np.uint8)
                    print(f"{predictions_dir}/{tile_name}.png")
                    cv2.imwrite(f"{predictions_dir}/{tile_name}.png", prob_mask)               

    # calculate mean iou
    eps = np.spacing(1.)
    intersection = np.stack(areas_intersection, axis=0).sum(axis=0).astype("float32")
    union = np.stack(areas_union, axis=0).sum(axis=0).astype("float32")
    iou = (intersection + eps) / (union + eps)
    mean_iou = iou.mean()
    print("Epoch {}, Valid Loss: {}, mIoU: {}".format(epoch, np.mean(val_losses), mean_iou))
    # loss and metrics averaged over all batches
    metrics = {'val_loss': np.mean(val_losses), 'miou': np.mean(mean_iou)}
   
    return metrics


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-name', type=str, default='unet_resnet50', help='String model name used for saving')
    arg('--encoder', type=str, default='resnet50', help='String model name used for saving')
    arg('--results-dir', type=str, default=RESULTS_DIR, help='Directory for saving model')
    arg('--data-dir', type=str, default=TRAIN_DIR, help='Directory for saving model')
    arg('--image-size', type=int, default=IMG_SIZE, help='Image size for training')
    arg('--batch-size', type=int, default=4, help='Batch size during training')
    arg('--num-workers', type=int, default=2, help='Number of workers for dataloader. Default = 4.')
    arg('--epochs', type=int, default=3, help='Epoch to run')
    arg('--lr', type=float, default=1e-3, help='Initial learning rate')
    arg('--checkpoint', type=str, default=None, help='Checkpoint filename with initial model weights')
    arg('--debug', type=bool, default=False)
    arg('--save-oof', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    set_seed(seed=1234)
   
    model = get_unet(encoder=args.encoder, in_channels = 4, num_classes = 1, activation = None) # +1 for background, 0 on the masks
    # load model weights to continue training
    if args.checkpoint is not None and args.checkpoint != '':
        load_model(model, args.checkpoint)

    train_runner(
        model=model,
        model_name=args.model_name,
        results_dir=args.results_dir,
        debug=args.debug,
        img_size=args.image_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_oof=False,
    )


if __name__ == "__main__":
    main()
