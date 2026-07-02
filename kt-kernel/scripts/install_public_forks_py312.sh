#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${1:-.venv-public-py312}"
SGLANG_URL="${SGLANG_URL:-git+https://github.com/PamelaSprin47685ghall/sglang.git@main#subdirectory=python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd -- "${REPO_ROOT}/../.." && pwd)/sglang-fork}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "missing interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -e "${VENV_DIR}" && "${FORCE_RECREATE:-0}" != "1" ]]; then
  echo "venv already exists: ${VENV_DIR}" >&2
  echo "set FORCE_RECREATE=1 to remove and recreate it" >&2
  exit 1
fi

if [[ "${FORCE_RECREATE:-0}" == "1" && -e "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

echo "creating venv with ${PYTHON_BIN} at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y sglang sglang-kt || true

echo "installing public sglang fork"
python -m pip install "sglang-kt @ ${SGLANG_URL}"

echo "installing local kt-kernel in editable mode"
cd "${REPO_ROOT}"
CPUINFER_USE_CUDA=0 \
CPUINFER_CPU_INSTRUCT=AVX2 \
CPUINFER_FORCE_REBUILD="${CPUINFER_FORCE_REBUILD:-0}" \
python -m pip install -e .

cat <<EOF

done.

activate:
  source ${VENV_DIR}/bin/activate

recommended smoke tests:
  python -m pytest test/per_commit/test_setup_avx2_variant_static.py -q
  python -m pytest "${SGLANG_REPO_ROOT}/test/srt/test_qwen35moe_name_mapping.py" "${SGLANG_REPO_ROOT}/test/srt/test_qwen35moe_transforms.py" -q

notes:
  - hwloc development files must already be installed.
  - override PYTHON_BIN, VENV_DIR, SGLANG_URL, SGLANG_REPO_ROOT, CPUINFER_FORCE_REBUILD as needed.
EOF
