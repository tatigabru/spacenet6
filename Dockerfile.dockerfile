FROM nvidia/cuda:9.0-runtime-ubuntu16.04

MAINTAINER Tati Gabru <tatigabru@gmail.com>

# System set up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libpng12-dev \
    libopencv-dev \
    libncurses-dev \
    libgl1 \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    protobuf-compiler \
	python3-rtree \
    unzip \
    openssh-server \
	screen \	 	  
	rsync \    
    wget \
    build-essential && \
  apt-get clean && \
  rm -rf /var/tmp /tmp /var/lib/apt/lists/*

# Get Anaconda
RUN curl -sSL -o installer.sh https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
    bash installer.sh -b -f && \
    source ~/.bashrc && \
    rm installer.sh && \ 
    export PATH="/root/anaconda3/bin:$PATH" && \
    conda  update -y conda      

RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
RUN pip install --upgrade pip && \
    pip install tqdm albumentations kaggle matplotlib numpy opencv-python shapely pretrainedmodels \
    pandas pytorch_toolbelt tifffile scikit-image scikit-learn segmentation-models-pytorch \
    tensorflow 


WORKDIR /work

COPY . /work/


#RUN chmod 777 train.sh
#RUN chmod 777 test.sh