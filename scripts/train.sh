#!/usr/bin/env bash

# 1. make folds
python -m src.folds.make_folds

# 2. create empty masks for images with no buildings
python -m src.utils.make_empty_masks

# 3. debug runners
python -m src.pretrain_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --image-size 224 --epochs 10 --lr 1e-2 --batch-size 8 --num-workers 2
python -m src.train_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --image-size 224 --debug True --epochs 2 --lr 1e-2 --batch-size 8 --num-workers 2 --save_oof True

# 4. pretrain model in RGB+ grayscale images

# 5. train model on SAR data

# 6. load checkpoints and plot val predictions

# 7. make test predictions


#sh predict_oof.sh $1
python -m src.pretrain_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --debug True --image-size 224 --epochs 2 --lr 1e-2 --batch-size 4 --num-workers 2 --save-oof False
Namespace(batch_size=4, checkpoint=None, data_dir='../data/train/AOI_11_Rotterdam/', debug=True, encoder='se_resnext101_32x4d', epochs=2, image_size=224, lr=0.01, model_name='unet_se_resnext101_32x4d', num_workers=2, results_dir='../output/', save_oof=True)