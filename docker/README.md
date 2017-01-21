# CaffeOnSpark Standalone Docker

Dockerfiles for both CPU and GPU builds are available in `standalone` folder. To use the CPU only version use the commands given. A GPU version of docker can be run using the command [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) instead of `docker` using the `standalone/gpu` folder. 

Dockerfiles for CPU build is provided in `standalone/cpu` folder. The image can be built by running:
```
docker build -t caffeonspark:cpu standalone/cpu
```
After the image is built, use `docker images` to validate.

## Launching CaffeOnSpark container
Hadoop and Spark are essential requirements for CaffeOnSpark. To ensure that both process runs flawless, we have included `standalone/cpu/config/bootstrap.sh` script which must be run everytime the container is started.

To launch a container running CaffeOnSpark please use:
```
docker run -it caffeonspark:cpu /etc/bootstrap.sh -bash
```

Now you have a working environment with CaffeOnSpark.

To verify installation, please follow [GetStarted_yarn](https://github.com/yahoo/CaffeOnSpark/wiki/GetStarted_yarn) guide from `Step 7`.
