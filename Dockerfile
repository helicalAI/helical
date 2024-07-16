FROM python:3.11

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
   

RUN pip install --upgrade --force-reinstall git+https://github.com/helicalAI/helical.git

# Define the entry point for the container
ENTRYPOINT ["/bin/bash"]