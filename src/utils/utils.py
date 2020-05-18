import os
import torch
import random
import numpy as np
from torch import nn
import json
from datetime import datetime
import logging


def set_seed(seed: int=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_optim(optimizer: torch.optim, checkpoint_path: str, device: torch.device):
    """
    Load optimizer to continuer training

        Args:
            optimizer: initialized optimizer
            checkpoint_path: path to the checkpoint
            device: device to send optimizer to (must be the same as in the model)
            
        Note: must be called after initializing the model    
    """  
    checkpoint = torch.load(checkpoint_path)    
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)    

    for param_group in optimizer.param_groups:
        print('learning_rate: {}'.format(param_group['lr']))    

    print('Loaded optimizer {} state from {}'.format(optimizer, checkpoint_path))    
    
    return optimizer


def save_ckpt(model: nn.Module, optimizer: torch.optim, checkpoint_path: str) -> dict:
    """
    Save model and optimizer checkpoint to continuer training
    """  
    torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                checkpoint_path
            )
    print("Saved model and optimizer state to {}".format(checkpoint_path))

def load_ckpt(checkpoint_path: str) -> dict:
    """
    Load checkpoint to continuer training
    """  
    checkpoint = torch.load(checkpoint_path)
        
    return checkpoint


def load_model(model: nn.Module, checkpoint_path: str) -> tuple:
    """Loads model weigths to continuer training
    """  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    return model, checkpoint
 
 
def write_event(log, step, **data):
    """Log event
    Source: https://github.com/ternaus/robot-surgery-segmentation/blob/master/utils.py
    Author: V. Iglovikov (ternaus)
    """
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image: np.array) -> bool:
    """Checks if image size divisible by 32
        Args:
            image: imput image/mask as a np.array        
        Returns:
            True if both height and width divisible by 32 and False otherwise.
    """
    image_height, image_width = image.shape[:2]
    return image_height % 32 == 0 and image_width % 32 == 0