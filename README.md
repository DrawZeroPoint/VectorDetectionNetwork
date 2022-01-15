# Welcome

This is the codebase of the [paper](https://ieeexplore.ieee.org/document/9526566) 

Z. Dong, Y. Gao, Y. Yan and F. Chen, "Vector Detection Network: An Application Study on Robots Reading Analog Meters in the Wild," in IEEE Transactions on Artificial Intelligence, vol. 2, no. 5, pp. 394-403, Oct. 2021, doi: 10.1109/TAI.2021.3105936.

We assume you download it with `git clone `, and the code folder `VDN/` (or other names you prefer) is located in `~`.

OS: Ubuntu 16.04 or 18.04

Language: Python 3.6+

Deep learning framework: PyTorch

# Prerequisites

## Hardware

Make sure the PC is with 8 GB or more GRAM.

Please make sure you have the correct version of the Nvidia driver installed, and that is compatible with your GPU card.

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
+-- demo.py  # Quick demo for a demonstration
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

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


The Pointer-10K dataset referred in our paper is publicly available for non-commercial usage. 
We adopt the CC BY-NC-SA 4.0 license, by which you can remix, adapt, and build upon this work non-commercially, 
as long as you credit this work and license your new creations under the identical terms.

Download the dataset here: [BaiduDisk](https://pan.baidu.com/s/1R1iZAqKJ2V656EW0RSM3Mg) password: p10k


# Basic Training

Before training the VDN model, (i) make sure you have the Pointer-10K dataset located in `~/Database/Done/pointer_10k`.
(ii) Download the pre-trained ResNet model `resnet50-19c8e357.pth` for parameter initialization from 
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

You can put your image into `VDN/data/demo`, and the algorithm will automatically find all images within the folder
and detect pointers in these images if any analog meters exist. Please note that VDN takes the image patches output
by a meter detector, with this the provided demo images should contain the whole dial face but not much background 
nor only a part of the meters.
 
The results of the demo will be output to automatically created folder `output/demo`.

# Experiments

You can use the `eval.py` script to perform the experiments conducted in the paper.
For example, to evaluate the performance of the default configuration (ResNet34 backbone
with 384x384 input size), just issue `python eval.py` in the root folder. Evaluations 
of ResNet18 and ResNet50 could be executed in the `master` branch,
whereas Res2Net50 is evaluated in individual branch `res2net50`.

The evaluation output could be found in `/VDN/output/eval-<backbone>`.

# Footnotes

If you use any contents in this work, please kindly consider citing:

```
@ARTICLE{
  9526566,  
  author={Dong, Zhipeng and Gao, Yi and Yan, Yunhui and Chen, Fei},  
  journal={IEEE Transactions on Artificial Intelligence},   
  title={Vector Detection Network: An Application Study on Robots Reading Analog Meters in the Wild},   
  year={2021},  
  volume={2},  
  number={5},  
  pages={394-403},  
  doi={10.1109/TAI.2021.3105936}
}
```

   [nv]: <https://github.com/NVIDIA/nvidia-docker>
