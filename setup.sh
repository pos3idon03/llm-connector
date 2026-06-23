#!/usr/bin/env bash
set -euo pipefail

# ── GPU detection ─────────────────────────────────────────────────────────────

GPU_VENDOR=""

if command -v nvidia-smi &>/dev/null; then
    GPU_VENDOR="nvidia"
elif command -v rocm-smi &>/dev/null || command -v rocminfo &>/dev/null; then
    GPU_VENDOR="amd"
else
    echo "ERROR: No supported GPU detected." >&2
    echo "       nvidia-smi not found and rocm-smi/rocminfo not found." >&2
    echo "       Install NVIDIA drivers (CUDA 12+) or AMD ROCm 6.2+ and re-run." >&2
    exit 1
fi

echo "Detected GPU vendor: ${GPU_VENDOR}"

# ── Driver version checks ─────────────────────────────────────────────────────

if [[ "${GPU_VENDOR}" == "nvidia" ]]; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || true)
    if [[ -z "${CUDA_VERSION}" ]]; then
        echo "ERROR: Could not read CUDA version from nvidia-smi." >&2
        exit 1
    fi
    if [[ "${CUDA_VERSION}" -lt 12 ]]; then
        echo "ERROR: CUDA 12+ required. Detected CUDA ${CUDA_VERSION}." >&2
        echo "       Update your NVIDIA drivers: https://developer.nvidia.com/cuda-downloads" >&2
        exit 1
    fi
    echo "CUDA ${CUDA_VERSION} detected — OK."
fi

if [[ "${GPU_VENDOR}" == "amd" ]]; then
    if ! rocminfo &>/dev/null; then
        echo "ERROR: rocminfo failed. Ensure ROCm 6.2 is installed correctly." >&2
        echo "       https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html" >&2
        exit 1
    fi
    ROCM_VERSION=$(rocminfo 2>/dev/null | grep -oP 'ROCm version: \K[0-9]+\.[0-9]+' | head -1 || echo "unknown")
    echo "ROCm ${ROCM_VERSION} detected — OK."
fi

# ── Python + Poetry checks ────────────────────────────────────────────────────

if ! command -v poetry &>/dev/null; then
    echo "ERROR: poetry not found. Install it first:" >&2
    echo "       curl -sSL https://install.python-poetry.org | python3 -" >&2
    exit 1
fi

POETRY_PYTHON=$(poetry env info --executable 2>/dev/null || echo "python3")
PYTHON_MINOR=$("${POETRY_PYTHON}" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
PYTHON_MAJOR=$("${POETRY_PYTHON}" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
if [[ "${PYTHON_MAJOR}" -ne 3 ]] || [[ "${PYTHON_MINOR}" -lt 11 ]] || [[ "${PYTHON_MINOR}" -ge 13 ]]; then
    CURRENT=$("${POETRY_PYTHON}" --version 2>&1 || echo "not found")
    echo "ERROR: Python 3.11 or 3.12 required. Detected: ${CURRENT}" >&2
    echo "" >&2
    echo "  Install Python 3.12:" >&2
    echo "    sudo apt install python3.12 python3.12-venv" >&2
    echo "" >&2
    echo "  Then tell Poetry to use it:" >&2
    echo "    poetry env use python3.12" >&2
    echo "" >&2
    echo "  Then re-run: bash setup.sh" >&2
    exit 1
fi

echo "Python ${PYTHON_MAJOR}.${PYTHON_MINOR} detected — OK."

if [[ "${GPU_VENDOR}" == "nvidia" ]]; then
    echo "Installing with NVIDIA/CUDA support..."
    poetry install --with nvidia

elif [[ "${GPU_VENDOR}" == "amd" ]]; then
    # Poetry can't resolve PyTorch ROCm's transitive deps (pytorch-triton-rocm is
    # not resolvable across sources), so the AMD path bypasses Poetry for GPU packages.
    echo "Installing base deps..."
    poetry install

    echo "Installing PyTorch ROCm 6.2..."
    poetry run pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

    echo "Installing vLLM..."
    poetry run pip install vllm
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env"
echo "  2. Edit .env — set MODEL_ID (and HF_TOKEN for gated models)"
echo "  3. poetry run llm-connector"
echo "  4. curl http://localhost:8000/health"
