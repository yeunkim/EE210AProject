#FROM ubuntu:16.04
FROM carndterm1_gpu

RUN apt-get -y update --fix-missing && apt-get install -y \
    build-essential \
    git \
    libxt6 \
    unzip \
    wget

RUN apt-get update && \
      apt-get -y install sudo

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo

RUN git clone https://github.com/torch/distro.git ~/torch --recursive && \
    cd ~/torch; bash install-deps && \
    ./install.sh

RUN cd / && git clone https://github.com/charlesq34/3dcnn.torch.git


#RUN apt-get purge nvidia-* && \
#RUN add-apt-repository ppa:graphics-drivers/ppa && apt-get install -y nvidia-375
RUN apt-get install -y lua5.2

#RUN wget https://luarocks.org/releases/luarocks-2.4.4.tar.gz && \
#    tar zxpf luarocks-2.4.4.tar.gz && \
#    cd luarocks-2.4.4 && \
#    ./configure;  make build

RUN apt-get install -y luarocks
ENV CUDA_BIN_PATH=/usr/local/cuda-8.0/

#RUN luarocks install torch
#RUN luarocks install cutorch
#RUN luarocks install nn
#
#RUN luarocks install cunn