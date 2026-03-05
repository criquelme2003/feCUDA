FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    python3-pybind11 \
    pybind11-dev \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /workspace

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir extern && cd extern && git clone --depth 1 https://github.com/dmlc/dlpack.git