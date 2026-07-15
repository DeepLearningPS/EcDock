#!/usr/bin/env bash
set -eo pipefail

# Usage:
#   bash install_ECDock_merged_fixed_v4.sh [ENV_NAME] [YAML_FILE] [REQ_FILE]
# Example:
#   bash install_ECDock_merged_fixed_v4.sh ecdock ECDock_merged_base.yaml ECDock_merged_pip_requirements.txt

ENV_NAME="${1:-ecdock}"
YAML_FILE="${2:-ECDock_merged_base.yaml}"
REQ_FILE="${3:-ECDock_merged_pip_requirements.txt}"

TORCH_VERSION="2.4.1"
TORCHVISION_VERSION="0.19.1"
TORCHAUDIO_VERSION="2.4.1"
TORCH_CUDA_TAG="cu124"
TORCH_CUDA_VERSION="12.4"
PYG_WHL_URL="https://data.pyg.org/whl/torch-2.4.0+cu124.html"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

if [ ! -f "${YAML_FILE}" ]; then
    echo "Error: ${YAML_FILE} not found in current directory."
    exit 1
fi

if [ ! -f "${REQ_FILE}" ]; then
    echo "Error: ${REQ_FILE} not found in current directory."
    exit 1
fi

# Do not enable `set -u` here.
# Some conda activation/deactivation scripts, especially MKL/BLAS hooks, read optional
# variables such as CONDA_MKL_INTERFACE_LAYER_BACKUP. With `set -u`, these hooks can
# abort with: unbound variable.

run_python_check_torch() {
python - <<PY
import sys
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if str(torch.version.cuda) != "${TORCH_CUDA_VERSION}":
    raise SystemExit(f"ERROR: expected torch CUDA ${TORCH_CUDA_VERSION}, got {torch.version.cuda}")
PY
}

is_protected_pkg() {
    # Return 0 if the package should NOT be installed from the generic requirements loop.
    # These packages are installed separately with fixed CUDA / --no-deps rules.
    local raw="$1"
    local pkg_name

    # Extract only the normalized package name from entries such as:
    #   torch==2.4.1, torch>=2.1, torch[extra]==x, torch ; python_version>="3.10"
    # This avoids putting '<' or '>' directly in a bash case pattern, which causes syntax errors.
    pkg_name="$(printf '%s' "${raw}"         | sed 's/[;].*$//'         | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'         | sed -E 's/[[:space:]]//g; s/\[.*\]//; s/(==|>=|<=|~=|!=|>|<|=).*$//'         | tr '[:upper:]_' '[:lower:]-')"

    case "${pkg_name}" in
        torch|torchvision|torchaudio|torchdata|torchmetrics|pytorch-lightning|lightning) return 0 ;;
        pyg-lib|torch-geometric|torch-scatter|torch-sparse|torch-cluster|torch-spline-conv) return 0 ;;
        dgl|oddt) return 0 ;;
        *) return 1 ;;
    esac
}


echo "[1/7] Preparing conda environment: ${ENV_NAME}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment ${ENV_NAME} already exists. Removing it first..."
    conda deactivate >/dev/null 2>&1 || true
    conda env remove -n "${ENV_NAME}" -y
fi

echo "Creating conda environment from ${YAML_FILE}"
conda env create -n "${ENV_NAME}" -f "${YAML_FILE}"

conda activate "${ENV_NAME}"

# Clean pip tooling first.
echo "[2/7] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

# Remove any accidentally inherited or preinstalled torch/PyG packages.
echo "[3/7] Installing fixed PyTorch ${TORCH_VERSION}+${TORCH_CUDA_TAG} stack"
python -m pip uninstall -y \
    torch torchvision torchaudio torchdata torchmetrics pytorch-lightning lightning \
    pyg-lib torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
    dgl oddt >/dev/null 2>&1 || true

python -m pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url "${PYTORCH_INDEX_URL}"

run_python_check_torch

# Install DGL without dependencies, so it cannot install/replace torch.
echo "[4/7] Installing DGL without touching PyTorch"
if ! python -m pip install --no-cache-dir --no-deps dgl==2.4.0; then
    echo "Warning: pip install dgl==2.4.0 --no-deps failed. Trying conda dgl CPU package with --no-deps..."
    if ! conda install -y -c dglteam/label/th24_cpu dgl --no-deps; then
        echo "Warning: DGL installation failed, skipped."
    fi
fi
run_python_check_torch



# Install generic requirements, but skip packages that can modify torch/CUDA stack.
echo "[5/7] Installing generic pip requirements, skipping protected torch/DGL/PyG/ODDT packages"
FAILED_PIP_PACKAGES=()
SKIPPED_PROTECTED_PACKAGES=()

while IFS= read -r pkg || [ -n "${pkg}" ]; do
    # Trim leading/trailing spaces for testing only; preserve original for pip.
    trimmed="$(echo "${pkg}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    # Skip empty lines and comments.
    [[ -z "${trimmed}" || "${trimmed}" =~ ^# ]] && continue

    if is_protected_pkg "${trimmed}"; then
        echo "Skipping protected package from requirements: ${trimmed}"
        SKIPPED_PROTECTED_PACKAGES+=("${trimmed}")
        continue
    fi

    echo "Installing: ${trimmed}"
    if ! python -m pip install --no-cache-dir "${trimmed}"; then
        echo "Warning: failed to install ${trimmed}, skipped."
        FAILED_PIP_PACKAGES+=("${trimmed}")
    fi

    # Guard against accidental torch replacement by transitive dependencies.
    run_python_check_torch

done < "${REQ_FILE}"

# Install protected torch-dependent Python packages without dependencies.
echo "[6/7] Installing protected torch-dependent packages with --no-deps"
python -m pip install --no-cache-dir --no-deps \
    torchdata==0.7.1 \
    torchmetrics==1.7.4 \
    pytorch-lightning==2.5.2 || echo "Warning: protected torch-dependent package installation failed."
run_python_check_torch

# ODDT 0.7 often fails under build isolation because its setup imports runtime deps.
echo "Installing ODDT with no build isolation"
python -m pip install --no-cache-dir six==1.17.0
if ! python -m pip install oddt==0.7 --no-build-isolation --no-cache-dir; then
    echo "Warning: oddt==0.7 installation failed, skipped."
    FAILED_PIP_PACKAGES+=("oddt==0.7")
fi
run_python_check_torch

# Install PyG extensions last, exactly matching torch 2.4 + cu124.
echo "[7/7] Installing PyG extensions matched to torch 2.4 + CUDA 12.4"
python -m pip uninstall -y pyg-lib torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv >/dev/null 2>&1 || true
python -m pip install --no-cache-dir --only-binary=:all: \
    pyg-lib==0.4.0+pt24cu124 \
    torch-scatter==2.1.2+pt24cu124 \
    torch-sparse==0.6.18+pt24cu124 \
    torch-cluster==1.6.3+pt24cu124 \
    torch-spline-conv==1.2.2+pt24cu124 \
    -f "${PYG_WHL_URL}"
pip install torch_geometric==2.6.1
run_python_check_torch

if [ ${#SKIPPED_PROTECTED_PACKAGES[@]} -ne 0 ]; then
    echo ""
    echo "Protected packages skipped from generic requirements and installed separately where needed:"
    printf '  - %s\n' "${SKIPPED_PROTECTED_PACKAGES[@]}"
fi

if [ ${#FAILED_PIP_PACKAGES[@]} -ne 0 ]; then
    echo ""
    echo "The following pip packages failed to install:"
    printf '  - %s\n' "${FAILED_PIP_PACKAGES[@]}"
else
    echo ""
    echo "All non-protected pip packages installed successfully."
fi

echo ""
echo "Final verification:"
python - <<'PY'
import importlib
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

modules = [
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
    "torch_geometric",
    "dgl",
    "pytorch_lightning",
    "torchdata",
    "torchmetrics",
    "oddt",
]
for name in modules:
    try:
        m = importlib.import_module(name)
        print(f"{name}: {getattr(m, '__version__', 'import ok')}")
    except Exception as e:
        print(f"{name} import failed: {e}")
PY

echo "Installation finished."
