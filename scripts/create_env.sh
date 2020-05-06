#!/bin/bash
# Install Anaconda3
curl -sSL -o installer.sh https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
bash installer.sh -b -f && \
source ~/.bashrc &&\
rm installer.sh
export PATH="/root/anaconda3/bin:$PATH" && \

conda update -y conda
conda create -y -n spacenet python=3.7
conda activate spacenet

conda install -y -n spacenet pytorch=0.4.1 cuda90 -c pytorch
pip install --upgrade pip
pip install -r requirements.txt