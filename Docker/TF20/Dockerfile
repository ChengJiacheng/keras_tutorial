# FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
FROM tensorflow/tensorflow:2.0.1-gpu-py3
# FROM continuumio/anaconda:2019.10-alpine
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev vim wget sudo psmisc locales cmake vim htop  graphviz libgraphviz-dev 
RUN apt-get update && apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 

# You can change what anaconda version you want at 
# https://repo.continuum.io/archive/
RUN wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
RUN bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p ~/anaconda  && rm Anaconda3-4.2.0-Linux-x86_64.sh
ENV PATH ~/anaconda/bin:$PATH


RUN locale-gen en_US.UTF-8
ADD https://raw.githubusercontent.com/ChengJiacheng/keras_tutorial/master/Docker/TF20/TF20.yml /tmp/environment.yml


RUN conda env create --file /tmp/environment.yml && echo "source activate TF20" >> ~/.bashrc
ENV PATH /opt/conda/envs/torch/bin:$PATH

ADD https://github.com/cdr/code-server/releases/download/3.0.1/code-server-3.0.1-linux-x86_64.tar.gz ./ 
RUN tar -xvzf code-server-3.0.1-linux-x86_64.tar.gz && rm -r code-server-3.0.1-linux-x86_64.tar.gz && code-server-3.0.1-linux-x86_64/code-server --install-extension ms-python.python

RUN apt-get install -y git
RUN git config --global user.name "ChengJiacheng" && git config --global user.email "chinging@foxmail.com" && git config --global credential.helper cache && git config --global credential.helper 'cache --timeout=3600'

# RUN pip uninstall --yes pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd