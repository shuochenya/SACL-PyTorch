#!/bin/bash
#$ -S /bin/bash
#$ -P giil
#$ -cwd
#$ -jc gpu-container_g8.72h
#$ -ac d=nvcr-pytorch-2208
 . /fefs/opt/dgx/env_set/nvcr-pytorch-2208.sh
 . /fefs/opt/dgx/env_set/nvcr-pytorch-2208-py3.sh

echo "Done: /fefs/opt/dgx/env_set/nvcr-pytorch"

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

echo "Done: MY_PROXY_URL"

python cl_train.py
