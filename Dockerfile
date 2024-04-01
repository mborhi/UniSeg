# Start with CUDA base image
# FROM nvidia/cuda:11.5.0-devel-ubuntu20.04
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV PYTHON_VERSION=3.8
ENV TORCH_VERSION=1.11.0
# ENV CUDNN_VERSION=8.3.2.44
ENV nnUNet_raw_data_base="/data"
ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
ENV RESULTS_FOLDER="/data/nnUNet_trained_models"

# Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch==${TORCH_VERSION}+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html

# Set CUDA paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

# Verify installation
RUN python3 -c "import torch; print('PyTorch version:', torch.__version__)"

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

COPY . /workspace
COPY ./Upstream/nnunet /usr/local/lib/python3.8/dist-packages/nnunet

# Define default command
CMD ["/bin/bash"]