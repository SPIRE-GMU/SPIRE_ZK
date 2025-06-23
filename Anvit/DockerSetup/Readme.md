# Docker Setup and Usage Guide

This guide provides instructions on how to build and run the **cuZK** project using the provided [Dockerfile](./Dockerfile).

## Prerequisites
- Ensure you have [Docker installed](https://docs.docker.com/get-docker/) on your system.
- Verify your installation by running:
  ```sh
  docker --version
  ```
- Verify you have `nvidia-container-runtime` installed
## Building the Docker Image
Open a terminal at the directory (current directory) containing the `Dockerfile` and run:
```sh
docker build -t cuda11.5-cuzk .
```
Replace `cuda11.5-cuzk` with a descriptive name for your image.

## Setup NVIDIA-CONTAINER-RUNTIME 
Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 
- Don't forget to configure the container to utilize gpu 
- To utilize the nvidia runtime as default edit the docker daemon configuration
  - Open the configuration file
    ```sh
    sudo nano /etc/docker/daemon.json
    ```
  - Update with appropriate configuration
  ```json
  {
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
  }
  ```
## Running the Docker Container
To start a container from the built image, run:
```sh
docker run -it --rm -v $PWD:/workspace cuda11.5-cuzk bash
```
- Replace `cuda11.5-cuzk` with the name of your image.
- `-v $PWD:/workspace` will put the present working directory inside the `/workspace` folder of the container. So, update it according to your need. 
- `bash` is the command that we'll execute when launching the container from the image. If you need to run anything else you can replace the `bash` with that command; e.g.: `nvidia-smi`, `nvcc --version` etc.
- If you have not set `nvidia` as default runtime use the following command to utilize the gpu.
```shell
docker run -it --rm --gpus all -v $PWD:/workspace cuda11.5-cuzk bash
```

## Running the cuZK project
Once you have launched the bash terminal of the docker, follow the cuZK instructions to run the project there.
- Don't forget to update the [Makefile](./cuZK/test/Makefile) with appropriate `sm_XY` version of the target GPU. For RTX 3090 ti, it's `sm_86`.