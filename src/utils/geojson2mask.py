"""
Script converts masks from json geojson format to binary masks.
"""
from pathlib import Path
import argparse
import re
from tqdm import tqdm
import solaris as sol
import geopandas as gpd
import cv2
from typing import Dict
from spacenet6.utils import get_id2_file_paths


def main():

    shape = (900, 900)    
    # Load meta
    meta = pd.read_csv(TRAIN_META)
    ids = meta.ImageId.values
    print(ids[:10])
    sample_ids = ids[:10]
    mask_path = f'{TRAIN_DIR}masks'
    
    path_to_geojson = f'{TRAIN_JSON}SN6_Train_AOI_11_Rotterdam_Buildings_{idx}.geojson'
    path_to_tiff = f'{TRAIN_RGB}SN6_Train_AOI_11_Rotterdam_PS-RGB_{idx}.tif'

    mask_path = args.data_path / "masks"
    mask_path.mkdir(exist_ok=True, parents=True)
    print(mask_path)
    geojson_path = args.data_path / "geojson_buildings"
    sar_path = args.data_path / "SAR-Intensity"
    geojson_id2path = get_id2_file_paths(geojson_path)
    sar_id2path = get_id2_file_paths(sar_path)

    for file_id, sar_path in tqdm(sar_id2path.items()):
        json_path = geojson_id2path[file_id]
        labels = gpd.read_file(json_path)
        mask = sol.vector.mask.footprint_mask(df=labels, reference_im=str(sar_path))
        print(mask_path / f"{file_id}.png")
        cv2.imwrite(str(mask_path / f"{file_id}.png"), mask)


if __name__ == "__main__":
    main()