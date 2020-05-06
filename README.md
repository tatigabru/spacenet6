# SpeceNet6 Cahllenge


## 

## ModelsUnet models with different encoders, https://github.com/qubvel/segmentation_models.pytorch, from [segmentation models libruary](https://github.com/qubvel/segmentation_models.pytorch) 


## Images preprocessing and augmentations
The original images were scaled to 512 x 512 px resolution. 
Progressive learning ...

The images were agmented using [albumentations libruary](https://albumentations.readthedocs.io/en/latest/index.html). The datasets with transforms are in ```src/datasets/```

## Training
All base models used were pre-trained on Imagenet dataset. 
Training script is in ```src/train_runner.py```

## Prepare environment 
1. Install anaconda
2. Run ```scripts/create_env.sh``` bash file to set up the conda environment

## Running the experiments 
Set up your own path ways to data and results_dir in configs.py.

Basic model: FPN ResNet101
Set paths in configs
Set IMS_SIZE in configs for transforms, use the same as in the args

For debugging, train running command is:
python -m src.train_runner --model-name "unet_resnet101" --encoder "resnet101" --debug True --image-size 224 --epochs 2 --lr 1e-2 --batch-size 4 --num-workers 2 --save-oof False

For training:
python -m src.train_runner --model-name "fpn_resnet101" --encoder "resnet101" --debug False --image-size 512 --epochs 50 --lr 1e-2 --batch-size 8 --num-workers 4 --save-oof False



