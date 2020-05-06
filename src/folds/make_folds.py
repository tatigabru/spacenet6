"""
Make folds
"""
import argparse
import copy
import json
import math
import os.path
import sys
from pathlib import Path
#sys.path.append("C:/Users/Tati/Documents/challenges/spacenet/progs/src")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .. configs import TRAIN_DIR, TRAIN_META
  

def create_stratified_folds(df: pd.DataFrame, nb_folds: int, save_dir: str, if_save: bool) -> pd.DataFrame:
    """
    Create folds
    Args: 
        df       : train meta dataframe       
        nb_folds : number of folds
        if_save  : boolean flag weather to save the folds
    Output: 
        df: train meta with splitted folds
    """
    df["fold"] = -1  # set all folds to -1 initially
    x = df.ImageId.unique()
    y = df.ImageId.count_values
    skf = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=42)
    # split folds
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        df.loc[test_index, "fold"] = fold
    # save dataframe with folds (optionally)
    if if_save:
        df.to_csv(f'{save_dir}/strat_folds.csv', index=False)
        
    return df


def create_folds(df: pd.DataFrame, nb_folds: int, save_dir: str, if_save: bool) -> pd.DataFrame:
    """
    Create folds
    Args: 
        df       : train meta dataframe       
        nb_folds : number of folds
        save_dir : directory, where to save folds
        if_save  : boolean flag weather to save the folds
    Output: 
        df: train meta with splitted folds
    """
    x = df.ImageId.unique()
    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=42)
    folds_df = pd.DataFrame()
    folds_df["ImageId"] = x
    folds_df["fold"] = -1  # set all folds to -1 initially
    
    # split folds
    for fold, (train_index, test_index) in enumerate(kf.split(x)):       
        x_test = x[test_index]        
        folds_df.loc[test_index, "fold"] = fold
    # save dataframe with folds
    if if_save:
        folds_df.to_csv(f'{save_dir}/folds.csv', index=False)
    
    return folds_df


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_META)
    print(df.head())
    x = df.ImageId.unique()
    print(x[:10])
    counts = df.groupby(df.ImageId).ImageId.value_counts
    print(counts)
    
    folds = create_folds(df, nb_folds = 5, save_dir = TRAIN_DIR, if_save = True)
    print(folds.head(20))
