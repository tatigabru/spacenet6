"""
https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
__author__ = Konstantin Lopuhin 
"""
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.wkt
import tifffile as tiff
from shapely.geometry import MultiPolygon, Polygon

import geopandas as gpd
import rasterio
from rasterio import features as rast_features

from ..configs import TRAIN_JSON, TRAIN_META, TRAIN_RGB, TRAIN_DIR

"""
Mask from geojson format

"""

def scale_coords(shape, Xmax, Ymin, point):
    """Scale the coordinates of a polygon into the image coordinates for a grid cell"""
    w,h = shape
    x,y = point[:,0], point[:,1]

    wp = float(w**2)/(w+1)
    xp = x/Xmax*wp

    hp = float(h**2)/(h+1)
    yp = y/Ymin*hp
    return np.concatenate([xp[:,None],yp[:,None]], axis=1)


def load_json(sample_id: str):    
    """ Load geojson file"""
    sh_fname = os.path.join(TRAIN_JSON, f'SN6_Train_AOI_11_Rotterdam_Buildings_{sample_id}.geojson')
    with open(sh_fname, 'r') as f:
        sh_json = json.load(f)
    return sh_json    


def get_polygons(sh_json): 
    # Scale the polygon coordinates to match the pixels
    polys = []
    for sh in sh_json['features']:
        geom = np.array(sh['geometry']['coordinates'][0])
        geom_fixed = scale_coords(shape, Xmax, Ymin, geom)
        pts = geom_fixed.astype(int)
        polys.append(pts)
    return polys    


def create_mask(polys, shape = (900, 900)):
    """ Create an empty mask and then fill in the polygons """
    mask = np.zeros(shape)
    cv2.fillPoly(mask, polys, 1)
    mask = mask.astype(bool)
    plt.imshow(mask)

    return mask


def read_mask(path_to_geojson, path_to_tiff):
    df = gpd.read_file(path_to_geojson)
    if df.empty:
        mask = np.zeros((900, 900), dtype=np.uint8)
        return mask
        
    feature_list = list(zip(df["geometry"], [255] * len(df)))
    mask = rast_features.rasterize(
        shapes=feature_list, 
        out_shape=(900, 900), 
        dtype = np.uint8,
    )
    return mask


if __name__ == "__main__":

    shape = (900, 900)    
    # Load meta
    meta = pd.read_csv(TRAIN_META)
    ids = meta.ImageId.values
    print(ids[:10])
    sample_ids = ids[:10]
    mask_path = f'{TRAIN_DIR}masks'

    for file_id in sample_ids:
        path_to_geojson = f'{TRAIN_JSON}SN6_Train_AOI_11_Rotterdam_Buildings_{file_id}.geojson'
        path_to_tiff = f'{TRAIN_RGB}SN6_Train_AOI_11_Rotterdam_PS-RGB_{file_id}.tif'
        mask = read_mask(path_to_geojson, path_to_tiff)
        print(f"{mask_path}/{file_id}.png")
        cv2.imwrite(f"{mask_path}/{file_id}.png", mask)


        
