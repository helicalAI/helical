FROM ubuntu:22.04

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get -y install build-essential \
        wget \
        git \
        curl \
        python3 \
        python3-pip

WORKDIR /usr/local/helical

COPY . /usr/local/helical
RUN pip install .

# Make the shell script executable
RUN chmod +x entrypoint.sh

# Define the entry point for the container
ENTRYPOINT ["/bin/bash"]