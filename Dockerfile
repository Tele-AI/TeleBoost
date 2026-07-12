# TeleBoost / TeleBoost training image, aligned with megatron-core 0.16.1.
#
# NGC tag is taken from the *official* Megatron-LM repo at the v0.16.1
# tag — `docker/.ngc_version.lts` pins it explicitly:
#   https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.1/docker/.ngc_version.lts
#       -> nvcr.io/nvidia/pytorch:25.09-py3   (LTS, stable testing matrix)
#   `.ngc_version.dev` uses 25.11-py3 (dev / bleeding edge)
#
# NGC 25.09 ships ABI-aligned: torch 2.9.0a0+nv25.09 / CUDA 13.0 /
# cuDNN / NCCL / cuBLAS / transformer_engine / apex (FusedAdam,
# FusedSGD, multi_tensor_applier). It does NOT bundle flash-attn
# (older NGC images like 24.10 did, current ones don't). We layer
# flash-attn 2/3 on top; both are built from source because upstream
# has no cu13/torch2.9 prebuilt wheels yet.
#
# Build:
#   docker build -t teleboost:mc0.16.1 -f Dockerfile .
#   # Hopper-only flash-attn 3 build adds ~45 min; skip with:
#   docker build --build-arg BUILD_FA3=0 -t teleboost:mc0.16.1 -f Dockerfile .
#
# Run (note: --shm-size 512G is required for distributed training):
#   docker run -it --gpus all --shm-size 512G \
#              -v /path/to/data:/data \
#              -v /path/to/code:/workspace \
#              teleboost:mc0.16.1

FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive

# ─── system packages ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        zsh curl wget htop ffmpeg \
        inetutils-ping net-tools zip tmux vim \
        cmake git gcc g++ make unzip rsync \
        openssh-server build-essential ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# ─── shell goodies ────────────────────────────────────────────────────
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
 && chsh -s "$(which zsh)" || true \
 && git clone --depth=1 https://github.com/zsh-users/zsh-autosuggestions \
        "${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions" \
 && sed -i 's/^ZSH_THEME="robbyrussell"/ZSH_THEME="ys"/' /root/.zshrc \
 && sed -i 's/^plugins=(git)/plugins=(git zsh-autosuggestions z)/' /root/.zshrc

# ─── pip index URL (default: official PyPI) ───────────────────────────
# China-internal builds may want to override to a local mirror, e.g.:
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple ...
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN rm -f /etc/pip.conf /etc/xdg/pip/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf /usr/pip.conf \
 && pip3 config set global.index-url "$PIP_INDEX_URL"

# ─── delete NGC-bundled opencv to avoid version conflicts ─────────────
# NGC ships an opencv variant whose cv2/typing module references newer
# attributes than what teleboost's pinned opencv-python==4.10 expects.
# Glob across Python versions (24.10=py3.10, 25.04=py3.10, 25.09=py3.12).
# Also wipe any opencv-contrib-python that might have been pulled by a
# transitive dep — it shadows opencv-python-headless and breaks cv2.dnn.
RUN find /usr/local/lib/python3*/dist-packages -maxdepth 2 -name cv2 -type d -exec rm -rf {} + 2>/dev/null \
 && pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless 2>/dev/null \
 && true

# ─── teleboost python deps (driven by requirements.txt) ────────────────
# Single source of truth lives in requirements.txt. DO NOT pip-install
# torch / transformer_engine / apex / NVIDIA CUDA libs here — NGC's
# pre-built versions are ABI-aligned with each other; replacing any one
# breaks the chain (apex's FusedAdam links against torch's CUDA libs,
# pip-installing apex from PyPI rebuilds it against the wrong ABI).
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ─── flash-attn 2 (source build — no cu13/torch2.9 wheel exists upstream)
# v2.8.3 is the latest 2.x release at time of writing. ~30–45 min build.
# MAX_JOBS=4 caps peak RAM during compile. FLASH_ATTENTION_FORCE_BUILD=TRUE
# bypasses the wheel-check fast-path that would otherwise fail.
ARG FA2_VERSION=2.8.3
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS}
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN pip3 install --no-cache-dir --no-build-isolation --no-deps \
      "flash-attn==${FA2_VERSION}"

# ─── flash-attn 3 (Hopper-only: H100 / H200 / H800; SM 9.0)
# transformer_engine's bf16 attention path looks up
# `flashattn_hopper.flash_attn_interface`. The pip package
# `flashattn-hopper` from the FA repo's hopper/ subdir provides the
# ops, but we also need to ship the python interface file alongside it.
# Skip this step (~45–60 min) on non-Hopper GPUs with --build-arg BUILD_FA3=0.
ARG BUILD_FA3=1
RUN if [ "${BUILD_FA3}" = "1" ]; then \
      pip3 install --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/Dao-AILab/flash-attention.git@v${FA2_VERSION}#subdirectory=hopper&egg=flashattn-hopper" && \
      PYPATH=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
      mkdir -p "$PYPATH/flashattn_hopper" && \
      curl -fsSL "https://raw.githubusercontent.com/Dao-AILab/flash-attention/v${FA2_VERSION}/hopper/flash_attn_interface.py" \
        -o "$PYPATH/flashattn_hopper/flash_attn_interface.py" ; \
    else \
      echo "BUILD_FA3=0: skipping flash-attn 3 (Hopper-only) build" ; \
    fi

# ─── megatron-LM (source build, matching verl v0.7.1's stable.vllm image)
# verl v0.7.1 pins NVIDIA/Megatron-LM at the ``core_v0.16.0`` tag and uses
# ISEEKYAN/mbridge for HF ↔ Megatron model bridging. Use the exact same
# pair so verl's TrainingWorker + checkpoint manager find the megatron
# symbols they expect at the patch level they were tested against.
RUN pip3 install --no-cache-dir --no-build-isolation --no-deps \
        "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.16.0" \
 && pip3 install --no-cache-dir -U \
        "git+https://github.com/ISEEKYAN/mbridge.git"

# ─── verl v0.7.1 (post-training framework — TrainingWorker for DPO) ────
# Same install pattern teleboost-grpo's v0.7.1 branch uses: pin upstream
# verl, vendor the missing ``experimental/reward_loop/router`` package
# from the GitHub tag (PyPI tarball misses it; harmless for DPO but
# imports break without it because ``init_workers`` references the
# router via ``RewardLoopManager``).
RUN pip3 install --no-cache-dir --no-deps \
        "verl @ git+https://github.com/volcengine/verl.git@v0.7.1" \
 && VERL_DIR=$(python3 -c "import verl, os; print(os.path.dirname(verl.__file__))") \
 && mkdir -p "$VERL_DIR/experimental/reward_loop/router" \
 && curl -fsSL -o "$VERL_DIR/experimental/reward_loop/router/naive_router.py" \
        "https://raw.githubusercontent.com/volcengine/verl/v0.7.1/verl/experimental/reward_loop/router/naive_router.py" \
 && touch "$VERL_DIR/experimental/reward_loop/router/__init__.py"

WORKDIR /workspace
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
