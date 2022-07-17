ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.12-py3
FROM $BASE_IMAGE

RUN pip install matplotlib
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
RUN pip install wandb==0.12.1