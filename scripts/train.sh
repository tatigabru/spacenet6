#!/usr/bin/env bash

python -m src.folds.make_folds
python -m src.convert_masks
python -m src.pretrain_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --image-size 224 --epochs 10 --lr 1e-2 --batch-size 8 --num-workers 2
python -m src.train_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --image-size 224 --epochs 10 --lr 1e-2 --batch-size 8 --num-workers 2 
#sh predict_oof.sh $1
python -m src.pretrain_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --debug True --image-size 224 --epochs 2 --lr 1e-2 --batch-size 4 --num-workers 2 --save-oof False
Namespace(batch_size=4, checkpoint=None, data_dir='../data/train/AOI_11_Rotterdam/', debug=True, encoder='se_resnext101_32x4d', epochs=2, image_size=224, lr=0.01, model_name='unet_se_resnext101_32x4d', num_workers=2, results_dir='../output/', save_oof=True)
# AWS IP
ec2-18-189-193-42.us-east-2.compute.amazonaws.com

#python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_0 d161_0 r34_0 sc50_0
#python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_1 d161_1 r34_1 sc50_1
#python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_2 d161_2 r34_2 sc50_2
#python ensemble.py --ensembling_dir oof/masks/ensemble --folds_dir oof/masks --dirs_to_ensemble d121_3 d161_3 r34_3 sc50_3
