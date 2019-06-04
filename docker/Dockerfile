ARG cuda_version=10.0
ARG cudnn_version=7.0
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

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
      python3.6-dbg \
      redis \
      && \
    rm -rf /var/lib/apt/lists/*

# Adding ppa for gis-unstable and installing gdal
RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update && apt-get install -y --no-install-recommends \
      gdal-bin=2.4.0+dfsg-1~bionic0 \
      python3-gdal=2.4.0+dfsg-1~bionic0

# Ensure locales configured correctly
RUN locale-gen en_US.UTF-8
ENV LC_ALL='en_US.utf8'

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# Install Python packages and keras
ENV NB_USER gcaffaratti
ENV NB_UID 10000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && echo "$NB_USER:$NB_USER" | chpasswd && adduser $NB_USER sudo && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER

ARG python_version=3.6

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/gdal
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/include/gdal

# PIP and Conda packages installation
RUN conda config --append channels conda-forge
RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install \
      sklearn_pandas \
      GDAL==2.4.0 \
      pygdal==2.4.0.4 \
      natsort \
      autokeras \
      redis \
      tensorflow-gpu \
      && \
    conda install \
      bcolz \
      h5py \
      matplotlib \
      seaborn \
      mkl \
      nose \
      notebook \
      Pillow \
      pandas \
      pydot \
      pygpu \
      pyyaml \
      numpy==1.16.1 \
      scikit-learn==0.20.2 \
      scipy==1.2.0 \
      six \
      theano \
      mkdocs \
      && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt

ADD theanorc /home/keras/.theanorc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /workspace

EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0
