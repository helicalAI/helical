FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get -y install build-essential \
        wget \
        git \
        curl \
        gcc \
        gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade helical
RUN pip install git+https://github.com/helicalAI/helical.git@main
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility 
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN pip install lightning wandb
# Define the entry point for the container
# ENTRYPOINT ["/bin/bash"]
