help:
        @cat Makefile

DATA?="${HOME}/CEPH"
WORK?="${HOME}/workspace"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.7
CUDA_VERSION?=10.0
CUDNN_VERSION?=7
UBUNTU_VERSION?=18.04
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
        docker build -t keras-foo --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) --build-arg ubuntu_version=$(UBUNTU_VERSION) -f $(DOCKER_FILE) .

bash: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --env KERAS_BACKEND=$(BACKEND) --cap-add=SYS_PTRACE keras-foo bash

ipython: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --env KERAS_BACKEND=$(BACKEND) keras-foo ipython

notebook: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --net=host --env KERAS_BACKEND=$(BACKEND) keras-foo

test: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --env KERAS_BACKEND=$(BACKEND) keras-foo py.test $(TEST)
