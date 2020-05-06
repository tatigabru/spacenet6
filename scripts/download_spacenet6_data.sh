#!/bin/bash
"""
Spacenet6 download
"""
CUR_DIR=$pwd
DATA_DIR_LOC=data

cd ..
cd ..
mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

if [ "$(ls -A $(pwd))" ]
then
    echo "$(pwd) not empty!"
else
    echo "$(pwd) is empty!"
    wget http://spacenet-dataset.s3.amazonaws.com/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_train.tar.gz
    wget http://spacenet-dataset.s3.amazonaws.com/spacenet/SN6_buildings/tarballs/SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz
    tar -xzf SN6_buildings_AOI_11_Rotterdam_train.tar.gz
    tar -xzf SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz
    rm SN6_buildings_AOI_11_Rotterdam_train.tar.gz
    rm SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz
fi

cd $CUR_DIR
echo $(pwd)