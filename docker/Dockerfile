FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive
# Core Linux Deps
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --fix-missing --no-install-recommends apt-utils \
	build-essential \
        curl \
	binutils \
	gdb \
        git \
	freeglut3 \
	freeglut3-dev \
	libxi-dev \
	libxmu-dev \
	gfortran \
        pkg-config \
	python-numpy \
	python-dev \
	python-setuptools \
	libboost-python-dev \
	libboost-thread-dev \
        pbzip2 \
        rsync \
        software-properties-common \
        libboost-all-dev \
        libopenblas-dev \ 
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
	libgraphicsmagick1-dev \
        libavresample-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev \
	libgraphicsmagick1-dev \
	libavcodec-dev \
	libgtk2.0-dev \
	liblapack-dev \
        liblapacke-dev \
	libswscale-dev \
	libcanberra-gtk-module \
        libboost-dev \
	libboost-all-dev \
        libeigen3-dev \
	wget \
        vim \
        qt5-default \
        unzip \
	zip \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*

ENV DEBIAN_FRONTEND noninteractive

ARG PYTHON=python3.6
ARG PIP=pip3

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

WORKDIR /
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN ls
RUN mv opencv-4.2.0 opencv
RUN mv opencv_contrib-4.2.0 opencv_contrib
RUN mkdir /opencv/build
WORKDIR /opencv/build

RUN apt update && apt install build-essential cmake pkg-config unzip yasm python3-numpy -y
RUN cmake -DBUILD_TIFF=ON \
        -DBUILD_opencv_java=OFF \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.0,7.5 \
        -DCUDA_ARCH_PTX=7.5 \
        -DENABLE_FAST_MATH=1 \
        -DCUDA_FAST_MATH=1 \
        -DWITH_CUBLAS=1 \
        -DENABLE_AVX=ON \
        -DWITH_OPENGL=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_IPP=ON \
        -DWITH_TBB=ON \
        -DWITH_EIGEN=ON \
        -DWITH_V4L=ON \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE="/usr/bin/python3" \
        -D PYTHON3_DEFAULT_EXECUTABLE="/usr/bin/python3" \
        -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
        -D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
        -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy as np; print(np.get_include())") \
        -DOPENCV_ENABLE_NONFREE=ON \
	-DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
        -DBUILD_EXAMPLES=ON \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 \
        -DWITH_QT=ON ..

RUN make -j10
RUN make install
RUN	rm /opencv.zip \
        && rm /opencv_contrib.zip \
		&& rm -rf /opencv \
        && rm -rf /opencv_contrib

# Prints "4.2.0"
RUN python3 -c "import cv2; print(cv2.__version__)"