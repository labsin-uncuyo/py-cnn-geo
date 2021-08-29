ARG cuda_version=10.0
ARG cudnn_version=7
ARG ubuntu_version=18.04
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel-ubuntu${ubuntu_version}

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      locales \
      sudo \
      nano \
      software-properties-common \
      python3-software-properties \
      libgdal-dev \
      gdb \
      python3.7 \
      redis \
      && \
    rm -rf /var/lib/apt/lists/*

# Adding ppa for gis-unstable and installing gdal
RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update && apt-get install -y --no-install-recommends \
      gdal-bin=3.0.4+dfsg-1~bionic0 \
      python3-gdal=3.0.4+dfsg-1~bionic0

# Ensure locales configured correctly
RUN locale-gen en_US.UTF-8
ENV LC_ALL='en_US.utf8'

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# Install Python packages and keras
ENV NB_USER gcaffaratti
ENV NB_UID 1002

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && echo "$NB_USER:$NB_USER" | chpasswd && adduser $NB_USER sudo && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER

ARG python_version=3.7

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/gdal
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/include/gdal

# PIP and Conda packages installation
RUN conda config --append channels conda-forge
RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install \
      natsort \
      redis \
      opencv-python==4.1.1.26 \
      scikit-learn==0.21.3 \
      sklearn-pandas==1.8.0 \
      && \
    conda install \
      tensorflow-gpu==2.0.0 \
      bcolz==1.2.1 \
      h5py==2.10.0 \
      matplotlib==3.3.4 \
      mkdocs==1.2.2 \
      mkl==2021.3.0 \
      nose==1.3.7 \
      notebook==6.4.3 \
      numpy==1.19.2 \
      scipy==1.5.2 \
      scikit-image==0.17.2 \
      pandas==1.1.5 \
      pillow==8.3.1 \
      pydot==1.4.1 \
      pygpu==0.7.6 \
      python  \
      pyyaml==5.4.1 \
      seaborn==0.11.2 \
      six \
      mkdocs \
      && \
#    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
#    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt

RUN pip install GDAL==3.0.4
    
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /workspace

EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0
