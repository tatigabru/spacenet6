#!/usr/bin/env bash
#Bash script to run tensorboard

CUR_DIR=$pwd
LOG_DIR= ../../output/tensorboard

pip install tensorboard 

tensorboard --logdir LOG_DIR --host 0.0.0.0 --port 6006