# syntax=docker/dockerfile:1.7

# =========================
# Stage 1: Builder
# =========================
FROM ros:noetic-ros-core-focal AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# APT / PIP / Conda 镜像
ARG UBUNTU_MIRROR=https://mirrors.bfsu.edu.cn/ubuntu
ARG PIP_INDEX_URL=https://mirrors.bfsu.edu.cn/pypi/web/simple

# 注意：
# BFSU 对 PyPI 的地址是 /pypi/web/simple，simple 不能省。
# Miniforge 安装包这里先保留 TUNA 的 github-release 镜像；
# 我没有找到可靠的 BFSU github-release/miniforge 路径。
ARG MINIFORGE_URL=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:${PATH} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=mirrors.bfsu.edu.cn \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

# 国内 APT 源：BFSU
RUN sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list && \
    sed -i "s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list && \
    sed -i "s|http://ports.ubuntu.com/ubuntu-ports|${UBUNTU_MIRROR}-ports|g" /etc/apt/sources.list || true

# 系统依赖。builder 阶段需要 python3-dev / portaudio19-dev 来编译 pyaudio
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        gnupg2 \
        lsb-release \
        sudo \
        ca-certificates \
        build-essential \
        bzip2 \
        python3-dev \
        portaudio19-dev \
        ros-noetic-cv-bridge \
        ros-noetic-apriltag-ros

# 安装 Miniforge。Miniforge 本身提供 conda / mamba 入口。
RUN curl -L "${MINIFORGE_URL}" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh && \
    conda --version && \
    (mamba --version || conda install -y mamba -c conda-forge)

# Conda 统一配置为 BFSU
# 这里的配置只在确实执行 conda/mamba install 时生效；
# 你的主环境仍然来自 myenv.tar.gz。
RUN cat > /root/.condarc <<'EOF'
channels:
  - conda-forge
  - defaults
show_channel_urls: true
channel_priority: strict
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  nvidia: https://mirrors.bfsu.edu.cn/anaconda/cloud
EOF

# pip 也写入全局配置，防止某些脚本没有继承 ENV
RUN mkdir -p /root/.pip && \
    cat > /root/.pip/pip.conf <<EOF
[global]
index-url = ${PIP_INDEX_URL}
trusted-host = mirrors.bfsu.edu.cn
disable-pip-version-check = true
EOF

WORKDIR /root/kuavo_data_challenge

# Conda 环境由独立 BuildKit context 提供，不进入源码 context。
# docker/build_classic.sh 会传入 --build-context classic_env=<directory>。
COPY --from=classic_env /myenv.tar.gz /tmp/myenv.tar.gz

# 解压主环境，并把通用 Python 依赖装进 ./myenv
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p ./myenv && \
    tar -xzf /tmp/myenv.tar.gz -C ./myenv && \
    rm -f /tmp/myenv.tar.gz && \
    source ./myenv/bin/activate && \
    conda-unpack && \
    python -m pip install \
        deprecated \
        pyaudio \
        kuavo_humanoid_sdk==1.3.3 \
        opencv-python==4.12.0.88 \
        opencv-python-headless==4.12.0.88 \
        numpy==2.2.6 \
        oss2 \
        requests && \
    conda clean -afy && \
    rm -rf ./myenv/lib/python*/site-packages/*/tests ./myenv/lib/python*/site-packages/*/test ./myenv/pkgs/*

# 最后才复制项目代码；模型权重和训练产物必须在运行时挂载。
COPY . .

# 项目 editable install 放最后。
# --no-deps 的意思是：不要每次让 pip 重新解析/下载依赖。
# 如果缺依赖，把依赖显式加到上一段 python -m pip install 里。
RUN --mount=type=cache,target=/root/.cache/pip \
    rm -f ./myenv.tar.gz && \
    source ./myenv/bin/activate && \
    python -m pip install -e . --no-deps --root-user-action=ignore && \
    if [ -d "./third_party/lerobot" ]; then \
        python -m pip install -e ./third_party/lerobot --no-deps --force-reinstall --root-user-action=ignore; \
    fi && \
    python -c "import lerobot, pathlib; p=pathlib.Path(lerobot.__file__).resolve(); assert str(p).startswith('/root/kuavo_data_challenge/third_party/lerobot/'), p; print('Verified container LeRobot:', p)" && \
    chmod -R a+rwX /root/kuavo_data_challenge/kuavo_deploy || true && \
    rm -rf \
        ./myenv/pkgs \
        ./myenv/conda-meta/history


# =========================
# Stage 2: Final
# =========================
FROM ros:noetic-ros-core-focal

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_MIRROR=https://mirrors.bfsu.edu.cn/ubuntu
ARG PIP_INDEX_URL=https://mirrors.bfsu.edu.cn/pypi/web/simple

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/root/kuavo_data_challenge/myenv/bin:/opt/conda/bin:${PATH} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=mirrors.bfsu.edu.cn \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /root/kuavo_data_challenge

RUN sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list && \
    sed -i "s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" /etc/apt/sources.list && \
    sed -i "s|http://ports.ubuntu.com/ubuntu-ports|${UBUNTU_MIRROR}-ports|g" /etc/apt/sources.list || true

# Final 阶段保守起见继续装 portaudio19-dev。
# 如果你想进一步瘦身，可尝试改为 libportaudio2。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-cv-bridge \
        ros-noetic-apriltag-ros \
        portaudio19-dev

# 保留 Miniforge/Mamba 保险工具链
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.condarc /root/.condarc
COPY --from=builder /root/.pip /root/.pip

# 复制主环境和项目代码。权重由运行时只读挂载提供。
COPY --from=builder /root/kuavo_data_challenge /root/kuavo_data_challenge

RUN chmod -R a+rwX /root/kuavo_data_challenge/kuavo_deploy || true && \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/kuavo_data_challenge/myenv/bin/activate" >> /root/.bashrc

CMD ["bash"]
