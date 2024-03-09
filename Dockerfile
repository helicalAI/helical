FROM ubuntu:22.04

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get -y install build-essential \
        wget \
        git \
        curl \
        python3-pip

# follow: https://stackoverflow.com/questions/68775869/message-support-for-password-authentication-was-removed
ARG GITHUB_TOKEN=
RUN pip install git+https://${GITHUB_TOKEN}@github.com/helicalAI/helical-package.git

WORKDIR /usr/local/helical

COPY requirements.txt /usr/local/helical
RUN pip install -r requirements.txt
COPY . /usr/local/helical

EXPOSE 8000

ENTRYPOINT ["/bin/bash"]
