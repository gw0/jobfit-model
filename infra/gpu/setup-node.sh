#!/usr/bin/env bash
# Lets a KinD node's containerd hand NVIDIA GPUs to pods (what NVIDIA's nvkind does);
# `make cluster-up GPU=1` runs it on the node, then applies the device plugin next to it.
#
# Host prerequisites, once, as root: the NVIDIA driver and nvidia-container-toolkit, with
#   nvidia-ctk runtime configure --runtime=docker --set-as-default
#   nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
#   systemctl restart docker
# so infra/kind-config.yaml's /var/run/nvidia-container-devices/all mount exposes every GPU
# to the node container.
#
# Usage: infra/gpu/setup-node.sh <kind-node-container>
set -euo pipefail

docker exec "$1" bash -euc '
  apt-get update -qq && apt-get install -y -qq curl gpg >/dev/null
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit >/dev/null
  nvidia-ctk runtime configure --runtime=containerd --set-as-default
  systemctl restart containerd
  # The driver files are bind-mounted in read-only; unmounting lets the in-node toolkit
  # mount them into each pod itself.
  umount -R /proc/driver/nvidia || true
'
