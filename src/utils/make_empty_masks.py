"""
convert masks to png
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List
from ..configs import TRAIN_MASKS, TRAIN_DIR, TRAIN_JSON


def masks2png(masks_dir: str, mask_path: str, file_ids: List[str]):
    """Convert numpy masks to png """
    for file_id in file_ids:
        try:
            mask = np.load(f'{masks_dir}{file_id}')
        except:
            mask = np.zeros((900, 900), np.uint8)    
        tile_name = file_id[37:-4]
        print(f"{mask_path}/{tile_name}.png")
        cv2.imwrite(f"{mask_path}/{tile_name}.png", mask)


def load_plot_mask(masks_dir: str, file_id: str):
    mask = np.load(f'{masks_dir}{file_id}')
    plt.imshow(mask)
    plt.show()


def make_empty_masks(masks_dir: str, json_dir: str, save_dir: str):
    """Create empty masks for missing json ids"""
    ids = os.listdir(masks_dir)
    id_names = [s[:-4] for s in ids]    
    jsons = os.listdir(TRAIN_JSON)
    json_ids = [s[:-8] for s in jsons]
    
    for json_id in json_ids:
        if json_id not in id_names:
            print(f'missing id: {json_id}')
            mask = np.zeros((900, 900), np.uint8)
            np.save(f'{save_dir}/{json_id}.npy', mask)


if __name__ == "__main__":

    shape = (900, 900)    
    # Load meta    
    ids = os.listdir(TRAIN_MASKS)
    tile_ids = [s[:-4] for s in ids]
    print(tile_ids[:10])
    sample_ids = ids[:10]
    mask_path = f'{TRAIN_DIR}masks_np'
    jsons = os.listdir(TRAIN_JSON)
    json_ids = [s[:-8] for s in jsons]
    print(json_ids[:10])

    make_empty_masks(TRAIN_MASKS, TRAIN_JSON, mask_path)

    #for sample in sample_ids:
    #    load_plot_mask(masks_dir, sample)
    #masks2png(masks_dir, mask_path, file_ids)



    