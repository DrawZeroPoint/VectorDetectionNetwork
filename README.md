# Welcome

This is the codebase of the paper titled "VDN: Pointer Points Detection Network Conquesting Pointer Detection in the Wild".
We assume you download it with `git clone `, and the code folder `/VDN` is located in `~`.

OS: Ubuntu 16.04 or 18.04
Lanuage: Python 3.6
Deep learning framework: Pytorch

# Prerequests

## Hardware

We use an ordinary PC to conduct the training and evaluation of our VDN model, and follows are the settings of the PC:

`CPU Intel i5-9600K, RAM 16GB, GPU NVIDIA GTX1070Ti, SSD 512GB.`

Since the GTX10 series GPU is no longer in production, you may turn to newer NVIDIA graphic cards. Please make sure you have
the right version of nvidia driver installed and that is compatible with your card.

## Software

We use Docker to ease the process of building the environment for running VDN. The installation of Docker can be done by:

```
wget -qO- https://get.docker.com/ | sh
systemctl enable docker.service
```

Meanwhile, you may also install [nvidia-docker][nv] plugin. To be brief, this is a quick guide:

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

Other than that, all software dependences can be handled within the Docker container. A detailed software dependences can be found in `VDN/Dockerfile`. For anonymity concern, we cannot provide our docker image, yet you may build one exactly as ours by:

```
cd ~/VDN
docker build --tag=vdn/vdn .
```

# File structure

This repo is organized as follows:

```
.
+-- compiled
+-- data
|   +-- demo
|   +-- results
+-- libs
|   +-- <PersonNameA>
|       +-- PersonNameA1.jpg
|       +-- ...
|   +-- <PersonNameB>
|       +-- PersonNameB1.jpg
|       +-- ...
+-- logs
|   +-- UCF101
|       +-- list
|           +-- classInd.txt  # 类别标签文件
|   +-- VOC2012
|   +-- ...
+-- modules
|   +-- Audio
|   +-- Other
|   +-- Thermal
|   +-- Visual
+-- utils
|   +-- identity_verification
|   +-- switch_recognition
|   +-- ...
+-- weights
+-- .dockerignore
+-- .gitignore
+-- add_aliases.sh
+-- Dockerfile
+-- LICENSE
+-- README.md
```

# Compile

We have provided a bash script `add_aliases.sh` to insert some handy bash scripts within the file `~/.bash_aliases`. It is recommended to
do so in `~/VDN`:

```
bash add_aliases.sh
source ~/.bash_aliases
```

Then, before training or testing VDN, you may run this to compile the code:

```
vdn_compile
```

# Train

Before training the VDN model, make sure you have the `Database` located in `~`.

```
# start the docker container
vdn_run

# train 
python train.py
```

# Run the demo

```
# start the docker container
vdn_run

# run the demo 
python demo.py
```

You can put your own image into `VDN/data/demo` and have fun detecting vectors (here, the image should contains a gauge with pointers :)



   [nv]: <https://github.com/NVIDIA/nvidia-docker>