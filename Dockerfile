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


# RUN mkdir /logs && chown 1000 /logs
# RUN mkdir /metaflow && chown 1000 /metaflow
# ENV HOME=/metaflow
# WORKDIR /metaflow
# USER 1000

# RUN pip install --upgrade helical
RUN pip install git+https://github.com/helicalAI/helical.git@main
RUN pip install metaflow simple-azure-blob-downloader azure-storage-blob azure-identity azure-keyvault-secrets

# Define the entry point for the container
# ENTRYPOINT ["/bin/bash"]