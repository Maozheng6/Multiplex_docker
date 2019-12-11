FROM mcr.microsoft.com/cntk/release
MAINTAINER Multiplex_semantic_seg

ENTRYPOINT []

RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get upgrade -y
RUN apt-get install -y git
RUN apt-get install vim -y
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get -y install openslide-tools wget
RUN apt-get -y install python-openslide
RUN pip3 install openslide-python
RUN apt-get -y install imagemagick
WORKDIR /root
RUN git clone https://github.com/Maozheng6/Multiplex_docker.git

WORKDIR /root/Multiplex_docker

CMD ["/bin/bash"]

