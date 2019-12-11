FROM mcr.microsoft.com/cntk/release
MAINTAINER Multiplex_semantic_seg

ENTRYPOINT []

RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils 
RUN apt-get upgrade -y 
RUN apt-get install -y git
RUN apt-get install vim -y
WORKDIR /root
RUN apt-get install -y python3-pip 
RUN pip3 install openslide-python
WORKDIR /root
#UN conda install -c bioconda openslide
RUN git clone https://github.com/Maozheng6/Multiplex_docker.git

WORKDIR /root/Multiplex_docker

CMD ["/bin/bash"]
