FROM python:3.10

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

# Define the entry point for the container
# ENTRYPOINT ["/bin/bash"]