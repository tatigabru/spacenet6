import rasterio
import numpy as np
import geopandas as gpd
from rasterio import features as rast_features


def read_mask(path_to_geojson, path_to_tiff):
    df = gpd.read_file(path_to_geojson)
    if df.empty:
        mask = np.zeros(rasterio.open(path_to_tiff).shape, dtype=np.uint8)
        return mask
        
    feature_list = list(zip(df["geometry"], [255] * len(df)))
    mask = rast_features.rasterize(
        shapes=feature_list, 
        out_shape=rasterio.open(path_to_tiff).shape, 
        transform=rasterio.open(path_to_tiff).transform,
    )
    return mask