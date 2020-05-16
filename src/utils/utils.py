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
    """Loads optimizer, epoch, to continuer training
    """  
    checkpoint = torch.load(checkpoint_path)    
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)    
    for param_group in optimizer.param_groups:
        print('learning_rate: {}'.format(param_group['lr']))    
    print('Loaded optimizer state from {}'.format(checkpoint_path))    
    print('Optimizer {}'.format(optimizer))

    return optimizer


def load_model(model: nn.Module, checkpoint_path: str):
    """Loads model weigths, epoch to continuer training
    """  
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']+1
    moiu = checkpoint['epoch']
    print('Loaded model from {}, epoch {}'.format(checkpoint_path, epoch))

    return model, epoch
 
 
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