# Docker Packaging Guide for Your Project

This guide explains how to build a Docker image that includes ROS Noetic + Miniforge + your project code + editable third-party packages.

---

## 1️⃣ Configure Docker Registry Mirrors (Optional)

Accessing Docker Hub from within China can be slow. You may use registry mirrors such as those provided by DaoCloud or other public accelerators.

1. Edit the Docker configuration file:

```bash
sudo vim /etc/docker/daemon.json
```

2. Replace its contents with the following:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "args": []
        }
    },
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.imgdb.de",
        "https://docker-0.unsee.tech",
        "https://docker.hlmirror.com",
        "https://docker.1ms.run",
        "https://func.ink",
        "https://lispy.org",
        "https://docker.xiaogenban1993.com"
    ]
}
```

3. Save the file and restart Docker:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl status docker
```

---

## 2️⃣ Package backend-specific Conda environments

The unified repository deliberately uses separate environments. Do not pack
one environment and reuse it for every backend.

| Image | Conda environment | Required runtime |
|---|---|---|
| Classic ACT / DP | `kdc_dev` | Python 3.10, PyTorch 2.7.1 + CUDA 12.6 |
| LingBot-VLA v1 | `kdc_vla` | Python 3.10, PyTorch 2.7.1 + CUDA 12.6, flash-attn |
| LingBot-VLA v2 | `lingbotvla_v2` | Python 3.12, PyTorch 2.8.0, flash-attn 2.8.3 |
| OpenPI | no archive | Built from the pinned `uv.lock` |

Install `conda-pack` once:

```bash
conda install -n base -c conda-forge conda-pack
```

Keep the large archives outside the Git/Docker source context. Each named
context must contain a file named exactly `myenv.tar.gz`:

```bash
mkdir -p \
  /mnt/pqssd/docker_envs/classic \
  /mnt/pqssd/docker_envs/lingbot-v1 \
  /mnt/pqssd/docker_envs/lingbot-v2

conda-pack \
  -n kdc_dev \
  --ignore-editable-packages \
  -o /mnt/pqssd/docker_envs/classic/myenv.tar.gz

conda-pack \
  -n kdc_vla \
  --ignore-editable-packages \
  -o /mnt/pqssd/docker_envs/lingbot-v1/myenv.tar.gz
```

`--ignore-editable-packages` is required because the unified repository,
LeRobot and LingBot are editable installs on the workstation. The Dockerfiles
reinstall their pinned source copies after unpacking.

Create v2 from the pinned v2 submodule. Do not reuse `lerobot_hil`: its Python
version is suitable, but its PyTorch version does not match LingBot-v2. Run the
wrapper on Ubuntu 22.04/glibc 2.35 (for example, the cloud builder), not on the
current Ubuntu 20.04/glibc 2.31 workstation. It first installs the exact Torch
stack, resolves/downloads the official wheel from that runtime tuple, verifies
the wheel's GLIBC symbol requirements, and then resumes upstream setup:

```bash
docker/create_lingbot_v2_env.sh

conda-pack \
  -n lingbotvla_v2 \
  --ignore-editable-packages \
  -o /mnt/pqssd/docker_envs/lingbot-v2/myenv.tar.gz
```

The v2 setup script verifies `torch.cuda.is_available()`, so the builder also
needs GPU passthrough. Driver 575.57.08/CUDA 12.9 is sufficient for the target
Torch CUDA runtime, but a new driver does not compensate for an old glibc.

Select flash-attn from the actual Torch runtime, not from the maximum CUDA
version printed by `nvidia-smi`. The resolver checks Python, the Torch
major/minor version, `torch.version.cuda`, CPU architecture and
`torch._C._GLIBCXX_USE_CXX11_ABI`, then requires an exact asset from the
official release. After download, it also scans the wheel's shared libraries
for their newest `GLIBC_*` symbol and compares it with the target system:

```bash
conda run -n kdc_vla \
  python docker/flash_attn_wheel.py --dry-run

conda run -n lingbotvla_v2 \
  python docker/flash_attn_wheel.py \
  --version 2.8.3 \
  --download-dir /mnt/pqssd/wheelhouse/flash-attn-v2.8.3
```

For the current v2 target, the required tuple is Python 3.12, Torch 2.8,
CUDA 12.x and CXX11 ABI `TRUE`, which resolves to:

```text
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Pass the downloaded file to `tools/create_train_env.sh --flash-attn-wheel`.
Never substitute an ABI `FALSE`, different Torch minor, or different CPython
tag wheel merely because pip accepts its filename.

The current LingBot-v1 environment is a deliberate exception: the official
2.8.3 cp310/Torch-2.7 wheel matches its visible tuple but requires GLIBC 2.32,
so it fails to import on this glibc 2.31 host. Keep using the locally proven
wheel:

```text
/mnt/pqssd/wheelhouse/flash_attn_2.7.0.post2_kdc_vla/flash_attn-2.7.0.post2-cp310-cp310-linux_x86_64.whl
sha256: b46d91bedacbd1ffd5f16f0ab3733bb07e1eeee013051fc3674a2f13cb8e8dde
```

It imports successfully with Python 3.10, Torch 2.7.1+cu126 and CXX11 ABI
`TRUE`. Re-evaluate it only when the v1 base system or Torch tuple changes.

After the environments are ready, the packaging commands above can also be
run safely through:

```bash
docker/package_conda_envs.sh classic
docker/package_conda_envs.sh lingbot-v1
docker/package_conda_envs.sh lingbot-v2
```

The wrapper refuses to overwrite existing archives and writes a companion
SHA-256 file. It also sets conda-pack's temporary directory under
`/mnt/pqssd/docker_envs/.tmp`; Codex/Desktop sandboxes often provide a
size-limited private temp directory even when `/tmp` itself has free space.

---

## 3️⃣ Build backend images

Use the repository scripts. They verify pinned submodule commits and pass each
archive through a BuildKit named context:

```bash
CLASSIC_ENV_ARCHIVE=/mnt/pqssd/docker_envs/classic/myenv.tar.gz \
  docker/build_classic.sh

LINGBOT_ENV_ARCHIVE=/mnt/pqssd/docker_envs/lingbot-v1/myenv.tar.gz \
  docker/build_lingbot.sh

LINGBOT_V2_ENV_ARCHIVE=/mnt/pqssd/docker_envs/lingbot-v2/myenv.tar.gz \
  docker/build_lingbot_v2.sh

docker/build_openpi.sh
```

Add `DRY_RUN=1` to any command to inspect it without building.

⚠️ Important:
- Checkpoints, pretrained weights, datasets and credentials must not enter the
  image. Mount them read-only at runtime.

### Below is a working **Dockerfile **：

[Dockerfile example](../Dockerfile)

### Key Features of This Dockerfile:

#### 1. Base Image
- Uses the official ROS Noetic image on Ubuntu 20.04: `ros:noetic-ros-base-focal`.

#### 2. Domestic Acceleration
- APT: Configured to use Tsinghua University mirrors.
- Conda: Channels set to Tsinghua mirrors.
- Pip: Uses Alibaba Cloud PyPI mirror.

#### 3. System Tools & ROS Packages
- Installs common utilities: `curl`, `wget`, `sudo`, `build-essential`, `bzip2`, etc.
- Installs ROS packages: `ros-noetic-ros-base`, `ros-noetic-cv-bridge`, `ros-noetic-apriltag-ros`(You may add other ROS dependencies as needed).

#### 4. Miniforge
- Installs Miniforge3 and sets up environment variables.

#### 5. Project Code & Conda Environment
- Sets working directory to `/root/kuavo_data_challenge`.
- Copies entire project source code.
- Extracts the packed Conda environment `myenv.tar.gz`.
- Runs `conda-unpack` to fix hardcoded paths.
- Reinstalls the project and third-party packages in editable mode.
- Cleans up test directories and cache to reduce final image size.

#### 6. Container Optimization
- Automatically activates the Conda environment by appending to `.bashrc`.
- Sets default command to `bash`.
- Uses multi-stage build: only the final runtime environment and source code are copied into the final image, excluding builder-stage temporary files—significantly reducing image size.

---

## 4️⃣ Export a Docker image as a TAR file

Export the image:

```bash
docker save -o kuavo-classic.tar kuavo-classic:latest
```

Replace the image and output names for LingBot-v1/v2 or OpenPI.

---

## 5️⃣ Run the Docker Container

### Below is a **sample shell script** for running the container:

[shell script example](run_with_gpu.sh)

### This script is used to start or create a Docker container:

- **Import Image**：
- **Check if the container exists**：
  - Exists → Start and Attach (`docker start -ai`)
  - Does not exist → Create a new container and start it (`docker run -it --gpus all --net=host ...`)
- **Set environment variable**：
  - ROS network configuration (`ROS_MASTER_URI`, `ROS_IP`)
- **Supports GPU containers**  

---

## 6️⃣ Precautions

For the competition test, you need to upload a compressed file, which should contain two files: one is kdc_v0.tar (the name can be changed), which is the compressed Docker image, and the other is the execution script run_with_gpu.sh (do not change this name).

You must ensure that the packaged Docker image can run the simulation test with the following code:

```bash
# Start docker
sh run_with_gpu.sh

# Start simulation automated testing
python kuavo_deploy/src/scripts/script_auto_test.py --task auto_test --config configs/deploy/kuavo_env.yaml

```
