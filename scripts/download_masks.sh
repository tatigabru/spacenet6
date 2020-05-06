
"""
Masks download
"""
CUR_DIR=$pwd
DATA_DIR_LOC=root/data/train/AOI_11_Rotterdam

cd ~
# create directory only if it does not exist
mkdir -p .kaggle
mv $CUR_DIR/kaggle.json ~.kaggle/ 
cd $DATA_DIR_LOC

kaggle datasets download blondinka/masks-spacenet
unzip masks-spacenet.zip
rm masks-spacenet.zip

cd $CUR_DIR
echo $(pwd)
