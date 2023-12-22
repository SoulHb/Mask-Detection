FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

RUN apt-get update

# Install Python 3.8
RUN apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip
# Copy  and install other requirements
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY . .
RUN pip3 install -r ./requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx