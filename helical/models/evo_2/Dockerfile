FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

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

RUN pip install git+https://github.com/helicalAI/helical.git
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility 
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN pip install lightning wandb
ENV CUDNN_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn

RUN pip install transformer-engine[pytorch]==1.13.0

RUN git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git
RUN cd evo2 && pip install .

##Overwriting the torch version from Vortex to make it work.
RUN pip install torchvision torch==2.6.0
RUN cd ../
RUN git clone https://github.com/helicalAI/helical.git && cd helical
# RUN cd helical/examples/run_models
