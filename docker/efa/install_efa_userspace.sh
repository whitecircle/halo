#!/usr/bin/env bash
# One EFA userspace for every Halo image: rdma-core, AWS libfabric and the aws-ofi-nccl plugin the
# NCCL weight-sync group needs to ride EFA between a trainer container and a rollout-server container.
#
# The whole stack is wire-sensitive, rdma-core included: the plugin probes libfabric for in-order
# RDMA writes at init and forces NCCL_PROTO=simple when the probe fails, and that probe answers
# through rdma-core's EFA provider (libefa). Two containers whose libefa builds answer differently
# form the NCCL group with different protocol tables and hang at the first collective, so every
# image installs the same three packages from this one script. Versions are pinned here and nowhere
# else:
#   * rdma-core and libfabric come from the AWS EFA installer release whose libfabric the NGC base
#     of the training image already bundles; its MOFED rdma-core is replaced;
#   * aws-ofi-nccl is built from a pinned commit against the NCCL wheel uv.lock pins, exposing the
#     GIN entry points DeepEP V2 needs for cross-node EP (the installer's own plugin package lacks them).
#
# Needs root, apt, curl, a C toolchain, a CUDA toolkit at /usr/local/cuda and the nvidia-nccl wheel
# installed.
set -euo pipefail

# x86_64 only, like the images' other architecture-specific steps; the installer's deb tree is per arch.
[ "$(uname -m)" = x86_64 ] || { echo "$0: x86_64 only, this is $(uname -m)" >&2; exit 1; }

EFA_INSTALLER_VERSION=1.46.0
EFA_INSTALLER_SHA256=8302bd7849afb95c903a875d7dcb6f85b3d7629e9a8b67d020031cfc6f4d0ee1
# The libfabric that installer ships, and that nvcr.io/nvidia/pytorch:26.03-py3 bundles.
LIBFABRIC_VERSION=2.3.1amzn4.0
AWS_OFI_NCCL_COMMIT=1f0a976f537f859d8ea70c6699f7d92ac89eb7af

EFA_PREFIX=/opt/amazon/efa
PLUGIN_PREFIX=/opt/amazon/aws-ofi-nccl
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [ "$#" -ne 0 ]; then
    echo "usage: $0 (no arguments)" >&2
    exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential autoconf automake libtool pkg-config git ca-certificates curl libhwloc-dev

. /etc/os-release
case "${VERSION_ID}" in
    22.04) deb_dir=UBUNTU2204 ;;
    24.04) deb_dir=UBUNTU2404 ;;
    *) echo "no EFA installer packages for ${PRETTY_NAME}" >&2; exit 1 ;;
esac
work=$(mktemp -d)
tarball="$work/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
curl -fsSL -o "$tarball" "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
echo "${EFA_INSTALLER_SHA256}  ${tarball}" | sha256sum -c -
tar -xzf "$tarball" -C "$work"
debs="$work/aws-efa-installer/DEBS/${deb_dir}/x86_64"
# Userspace only: the verbs libraries and the EFA provider (rdma-core) plus libfabric. The kernel
# module, MPI, and the installer's own NCCL plugin (no GIN) are not installed. --allow-downgrades so
# the pin also wins over a base whose rdma-core has moved past it: the builds must match across images.
apt-get install -y --no-install-recommends --allow-downgrades \
    "$debs"/rdma-core/libibverbs1_*.deb \
    "$debs"/rdma-core/ibverbs-providers_*.deb \
    "$debs"/rdma-core/librdmacm1_*.deb \
    "$debs"/libfabric1-aws_${LIBFABRIC_VERSION}_amd64.deb \
    "$debs"/libfabric-aws-dev_${LIBFABRIC_VERSION}_amd64.deb \
    "$debs"/libfabric-aws-bin_${LIBFABRIC_VERSION}_amd64.deb
rm -rf /var/lib/apt/lists/* "$work"

test -x "$EFA_PREFIX/bin/fi_info" || { echo "fi_info missing under $EFA_PREFIX" >&2; exit 1; }
"$EFA_PREFIX/bin/fi_info" --version | grep -F "libfabric: ${LIBFABRIC_VERSION}" \
    || { echo "libfabric under $EFA_PREFIX is not ${LIBFABRIC_VERSION}:" >&2; "$EFA_PREFIX/bin/fi_info" --version >&2; exit 1; }

nccl_home=$(python3 -c 'import nvidia.nccl; print(nvidia.nccl.__path__[0])')
test -f "$nccl_home/include/nccl.h" || { echo "nccl.h missing under $nccl_home" >&2; exit 1; }

src=$(mktemp -d)
git clone https://github.com/aws/aws-ofi-nccl.git "$src/aws-ofi-nccl"
cd "$src/aws-ofi-nccl"
git checkout "$AWS_OFI_NCCL_COMMIT"
git submodule update --init --recursive
./autogen.sh
# The rpath pins the plugin to the AWS libfabric even on a base that also ships a distro libfabric.
./configure --prefix="$PLUGIN_PREFIX" \
    --with-libfabric="$EFA_PREFIX" \
    --with-cuda="$CUDA_HOME" \
    --with-nccl="$nccl_home" \
    --enable-platform-aws \
    --disable-tests \
    LDFLAGS="-Wl,-rpath,$EFA_PREFIX/lib"
make -j"$(nproc)"
make install
cd /
rm -rf "$src"

# NCCL loads the GIN plugin from this name, resolved through the loader's default directory (a
# symlink under the plugin prefix would be cached under the target's own SONAME instead). The net
# plugin is found as libnccl-net-ofi.so, and make install also placed it under NCCL's default plugin
# name libnccl-net.so, so it is tried on every host and yields to NCCL's built-in transports where
# libfabric finds no provider.
ln -sf "$PLUGIN_PREFIX/lib/libnccl-net-ofi.so" /usr/lib/x86_64-linux-gnu/libnccl-gin.so
printf '%s\n' "$EFA_PREFIX/lib" > /etc/ld.so.conf.d/efa.conf
printf '%s\n' "$PLUGIN_PREFIX/lib" > /etc/ld.so.conf.d/aws-ofi-nccl.conf
ldconfig

plugin="$PLUGIN_PREFIX/lib/libnccl-net-ofi.so"
short=${AWS_OFI_NCCL_COMMIT:0:7}
# grep reads the whole stream: a quiet-mode early exit sends `strings` a SIGPIPE, which pipefail
# reports as a failure.
strings "$plugin" | grep -F "git-${short}" >/dev/null \
    || { echo "plugin does not identify as git-${short}" >&2; exit 1; }
# The net and GIN plugin API versions the pinned NCCL loads.
for sym in ncclNetPlugin_v12 ncclGinPlugin_v13; do
    nm -D "$plugin" | grep " D ${sym}$" >/dev/null || { echo "plugin lacks ${sym}" >&2; exit 1; }
done
ldd "$plugin" | grep -F "libfabric.so.1 => $EFA_PREFIX/lib/" >/dev/null \
    || { echo "plugin resolves a libfabric outside $EFA_PREFIX:" >&2; ldd "$plugin" >&2; exit 1; }
if ldd "$plugin" | grep "not found" >/dev/null; then echo "plugin has unresolved libraries:" >&2; ldd "$plugin" >&2; exit 1; fi
# awk reads the whole listing: an early exit would hand ldconfig a SIGPIPE, which pipefail reports.
libefa=$(ldconfig -p | awk '/libefa\.so\.1 / && !found {found=$NF} END{print found}')
test -n "$libefa" || { echo "libefa.so.1 (the rdma-core EFA provider) is not in the loader cache" >&2; exit 1; }
echo "EFA userspace: $(basename "$(readlink -f "$libefa")"), libfabric ${LIBFABRIC_VERSION}, aws-ofi-nccl git-${short} (net v12, GIN v13)"
