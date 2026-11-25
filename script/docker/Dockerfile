###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Build environment for aiehlc: host toolchain + the packages LLVM/MLIR,
# aiehlc, aie-rt and Vitis (mounted from the host at /proj) need.
#   docker build -t aiehlc-dev --build-arg UID=$(id -u) --build-arg GID=$(id -g) \
#       --build-arg USERNAME=$(id -un) script/docker
FROM ubuntu:22.04
ARG USERNAME=aiehlc
ARG UID=1000
ARG GID=1000
ENV DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ gcc make cmake ninja-build ccache git ca-certificates curl wget \
        unzip xz-utils file rsync python3 python3-pip python3-venv python3-dev \
        libboost-all-dev libncurses5-dev libncurses5 libtinfo5 libtinfo-dev zlib1g-dev libzstd-dev libxml2-dev \
        binutils lld locales tzdata sudo openssh-client sshpass expect telnet iputils-ping net-tools \
        libx11-6 libxext6 libxrender1 libxtst6 libxi6 libgtk-3-0 libglib2.0-0 libnss3 libasound2 \
        libc6-dev-i386 lib32stdc++6 graphviz bc gawk device-tree-compiler \
    && locale-gen en_US.UTF-8 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy pyyaml pexpect paramiko
RUN (getent group ${GID} || groupadd -g ${GID} ${USERNAME}) \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
WORKDIR /proj
