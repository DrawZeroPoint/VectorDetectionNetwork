# Welcome

This is the codebase of the paper titled "Vector Detection Network: Pointer Detector for Robots Reading Analog Meters in the Wild".
We assume you download it with `git clone `, and the code folder `/VDN` is located in `~`.

OS: Ubuntu 16.04 or 18.04
Language: Python 3.6+
Deep learning framework: PyTorch

# Prerequisites

## Hardware

We use an PC to conduct the training and evaluation of the VDN model, and the hardware settings are:

`CPU Intel i5-9600K, RAM 16GB, GPU NVIDIA GTX1070Ti, SSD 512GB.`

Since the GTX10 series GPU is no longer in production, you could turn to newer NVIDIA graphic cards.
Please make sure you have the right version of nvidia driver installed and that is compatible with your card.

## Software

We use Docker to ease the process of building the environment for running VDN. The installation of Docker can be done by:

```
wget -qO- https://get.docker.com/ | sh
systemctl enable docker.service
```

Meanwhile, you should also install [nvidia-docker][nv] plugin. To be brief, this is a quick guide:

```
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

Other than that, all software dependence can be handled within the Docker container.
A detailed software dependence list could be found in `VDN/Dockerfile`.
For anonymity concern, we do not provide our docker image, yet you may build one exactly as ours by:

```
cd ~/VDN
docker build --tag=vdn/vdn .
```

# File structure

This repo is organized as follows:

```
.
+-- cfgs  # The configurations of different network architectures
|
+-- compiled  # Compiled third-party libraries
|
+-- data
|   +-- demo
|   +-- result
|
+-- libs  # VDN libraries
|
+-- modules  # The VDN class
|
+-- utils  # Some handy utilities
|
+-- weights
|   +-- pretrained
|
+-- .dockerignore
+-- .gitignore
+-- add_aliases.sh  # Bash script for adding Docker shortcuts to bash_aliaes
+-- Dockerfile
+-- LICENSE
+-- README.md
|
+-- demo.py  # Quick demo for demonstration
+-- train.py  # Training script of VDN
+-- test.py  # Evaluation and experiments for VDN

```

# Compile

We have provided a bash script `add_aliases.sh`  to insert some handy bash scripts within the file `~/.bash_aliases`.
It is recommended to do so in the root folder of this project:

```
bash add_aliases.sh
source ~/.bash_aliases
```

Then, before training or testing VDN, run this to compile the code:

```
vdn_compile
```

# The Pointer-10K dataset

The Pointer-10K dataset referred in our paper is public available for non commercial usage. If you are interested 
in the data, please contact us via email. The address will be released afterwards. 


# Basic Training

Before training the VDN model, (i) make sure you have the Pointer-10K dataset located in `~/Database/Done/pointer_10k`.
(ii) Download the pretrained resnet model `resnet50-19c8e357.pth` for parameter initialization from 
[torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
and put it in `weights/pretrained/` (you may need to create the path manually). 

```
# start the docker container
vdn_run

# train 
python train.py
```

# Run the demo

To run the demo, put the trained model named as `vdn_best.pth.tar` into `weights/`, and
run the code below

```
# start the docker container
vdn_run

# run the demo within the container
python demo.py
```

We provide the model trained by us: [download]()

You can put your own image into `VDN/data/demo` and the algorithm will automatically find all images within the folder
and detect pointers in these images if any analog meters exists. Please note that VDN takes the image patches output
by a meter detector, hereby the provided demo images should contain the whole dial face but not much background nor 
only a part of the meters.
 
Results of the demo will be output to automatically created folder `output/demo`.


   [nv]: <https://github.com/NVIDIA/nvidia-docker>