help:
        @cat Makefile

DATA?="${HOME}/Data"
WORK?="${HOME}/workspace"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=10.0
CUDNN_VERSION?=7
TEST=tests/
SRC?=$(shell dirname `pwd`)

build:
        docker build -t keras-foo --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) -f $(DOCKER_FILE) .

bash: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --env KERAS_BACKEND=$(BACKEND) --cap-add=SYS_PTRACE keras-foo bash

ipython: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --env KERAS_BACKEND=$(BACKEND) keras-foo ipython

notebook: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --net=host --env KERAS_BACKEND=$(BACKEND) keras-foo

test: build
        $(DOCKER) run -it -v $(SRC):/src/workspace -v $(DATA):/data -v $(WORK):/workspace --env KERAS_BACKEND=$(BACKEND) keras-foo py.test $(TEST)
