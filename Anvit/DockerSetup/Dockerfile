FROM nvidia/cuda:11.5.2-base-ubuntu20.04

# Set non-interactive mode for apt and timezone
ENV DEBIAN_FRONTEND=noninteractive TZ=America/New_York

WORKDIR /workspace

# Set timezone manually to prevent tzdata prompt
RUN ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime && \
    echo "America/New_York" > /etc/timezone && \
    apt-get update && apt-get install -y tzdata

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    python3 \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-5 \
    cuda-nvrtc-11-5 && \
    rm -rf /var/lib/apt/lists/* \

# Set up the environment for CUDA and C++
ENV PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install software-properties-common (provides add-apt-repository) and lsb-release
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && apt-get update

# Install GCC-12, G++-12, cmake, and git (build-essential already installed)
RUN apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake && \
    rm -rf /var/lib/apt/lists/* \

# Optionally, set CC and CXX environment variables
ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++
# Install rustup non-interactively with default options
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Add rustup (cargo) to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get -y update && apt-get install -y nano
RUN apt-get install -y libgmp-dev

# Set working directory for your projects
WORKDIR /workspace
