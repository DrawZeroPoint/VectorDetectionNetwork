# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM ufoym/deepo:pytorch-py36-cu100

# ==================================================================
# Set the working directory
# ------------------------------------------------------------------
WORKDIR /VDN

# =================================================================
# Define environment variable
# -----------------------------------------------------------------
ENV LANG C.UTF-8
ENV PYTHONPATH="$PYTHONPATH:/VDN"

# ==================================================================
# Make port available to the world outside this container
# ------------------------------------------------------------------
EXPOSE 80


# ==================================================================
# Install dependences of OpenCV
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ==================================================================
# Install useful tools
# ------------------------------------------------------------------
RUN python -m pip --no-cache-dir install --upgrade \
        opencv-python==3.4.2.16 \
        opencv-contrib-python==3.4.2.16 \
        torchsnooper \
        tensorboard \
        tqdm \
        terminaltables \
        pycocotools \
        imageio \
        easydict \
        pathlib2 \
        scikit-image \
        imutils \
