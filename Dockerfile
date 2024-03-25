# Start with CUDA base image
# FROM nvidia/cuda:11.5.0-devel-ubuntu20.04
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set environment variables
ENV PYTHON_VERSION=3.8
ENV TORCH_VERSION=1.11.0
ENV CUDNN_VERSION=8.3.2.44
ENV nnUNet_raw_data_base="/data"
ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/data/nnUNet_trained_models"

# # Install Python and pip
# RUN apt-get update && \
#     apt-get install -y \
#     rm get-pip.py

# # Install PyTorch with CUDA support
# RUN pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION} -f https://download.pytorch.org/whl/torch_stable.html

# # Install CuDNN
# RUN apt-get install -y libcudnn8=${CUDNN_VERSION}-1+cuda${CUDA_VERSION} \
#                        libcudnn8-dev=${CUDNN_VERSION}-1+cuda${CUDA_VERSION}

# Install nnUnet
RUN pip install nnUnet && \
    pip install nibabel && \
    pip install numpy && \
    pip install tqdm && \
    pip install SimpleITK

RUN mkdir /output
RUN mkdir /data
# Set default working directory
WORKDIR /workspace