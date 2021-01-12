FROM ubuntu:21.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update \
    && apt install -y python3-dev python3-pip

RUN pip install virtualenv

RUN mkdir root/src

RUN virtualenv -p python3 root/src/.env
RUN /bin/bash -c "source root/src/.env/bin/activate"

COPY . root/src

RUN pip install -r root/src/requirements.txt
