FROM python:3.11

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get -y install build-essential \
        wget \
        git \
        curl \
        gcc \
        gfortran \
        openblas-dev 

# WORKDIR /usr/local/helical

# COPY . /usr/local/helical
# RUN pip install .

RUN pip install --upgrade --force-reinstall git+https://github.com/helicalAI/helical.git
# RUN python3 -m pip install --index-url https://test.pypi.org/simple/ helical

# Make the shell script executable
RUN chmod +x entrypoint.sh

# Define the entry point for the container
ENTRYPOINT ["/bin/bash"]