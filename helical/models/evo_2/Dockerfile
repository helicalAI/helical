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

# RUN conda install nvidia::cuda-toolkit --yes
# ENV CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/include

# ENV CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/include:$CPLUS_INCLUDE_PATH
# RUN pip install --upgrade helical
RUN pip install git+https://github.com/helicalAI/helical.git@evo-2
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility 
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN pip install lightning wandb
ENV CUDNN_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn
# ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib
# ENV PATH=/usr/local/cuda/bin:$PATH
# ENV CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/include

# ENV CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/include:$CPLUS_INCLUDE_PATH
RUN pip install transformer-engine[pytorch]
# # Edit the pyproject.toml line in the vortex repo with the torch version to:     "torch==2.6.0",
# RUN git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
RUN git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git
RUN cd evo2 && pip install .

RUN git clone https://github.com/helicalAI/helical.git && cd helical && git checkout evo-2
# # Define the entry point for the container
# # ENTRYPOINT ["/bin/bash"]
