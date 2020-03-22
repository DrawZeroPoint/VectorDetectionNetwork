#!/bin/bash

# compile
echo "alias vdn_compile='docker run --runtime=nvidia -v ~/VectorDetectionNetwork:/VDN -v /dev:/dev \
--ipc=host --rm vdn/vdn:latest bash -c \"cd /VDN/compiled && make all\"'" >> ~/.bash_aliases

# run the container
echo "alias vdn_run='docker run --runtime=nvidia -v ~/VectorDetectionNetwork:/VDN -v ~/Database:/Database \
-v /dev:/dev --ipc=host -it --rm vdn/vdn:latest bash'" >> ~/.bash_aliases
