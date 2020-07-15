# FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
FROM tensorflow/tensorflow:2.0.1-gpu-py3

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev vim wget sudo psmisc locales cmake vim htop  

RUN apt-get update && apt-get install -y graphviz libgraphviz-dev

RUN locale-gen en_US.UTF-8
ADD https://gitlab.nautilus.optiputer.net/jiacheng/nautilus/-/raw/master/yaml/torch15.yml /tmp/environment.yml

COPY tf20.txt /tmp/environment.yml

RUN conda env create --file /tmp/environment.yml && conda init bash && echo "source activate TF20" >> ~/.bashrc
ENV PATH /opt/conda/envs/torch/bin:$PATH

# ADD https://github.com/cdr/code-server/releases/download/3.0.1/code-server-3.0.1-linux-x86_64.tar.gz ./ 
# RUN tar -xvzf code-server-3.0.1-linux-x86_64.tar.gz && rm -r code-server-3.0.1-linux-x86_64.tar.gz && code-server-3.0.1-linux-x86_64/code-server --install-extension ms-python.python
 
# RUN git config --global user.name "ChengJiacheng" && git config --global user.email "chinging@foxmail.com" && git config --global credential.helper cache && git config --global credential.helper 'cache --timeout=3600'

# RUN pip uninstall --yes pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd