#!/bin/bash

# compile
echo "alias vdn_compile='docker run --runtime=nvidia -v ~/VDN:/VDN -v /dev:/dev --ipc=host --rm kilox/lingxi:latest bash -c \"cd compiled && make all\"'" >> ~/.bash_aliases

# run the container
echo "alias vdn_run='docker run --runtime=nvidia -v ~/VDN:/VDN -v ~/Database:/Database -v /dev:/dev --ipc=host -it --rm kilox/lingxi:latest bash'" >> ~/.bash_aliases
