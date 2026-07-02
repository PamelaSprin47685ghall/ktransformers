# shellcheck shell=bash
# Source from ornith/*.sh — sets ORNITH_ROOT, PY, PYTHONPATH, KT_PYTHON.
_ornith_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
KT_ROOT="$(cd -- "${_ornith_script_dir}/../.." && pwd)"
ORNITH_ROOT="${ORNITH_ROOT:-$(cd -- "${KT_ROOT}/../.." && pwd)}"
HF_TEMPLATE_DEFAULT="${KT_ROOT}/fixtures/Ornith-1.0-35B-hf"
if [[ ! -f "${HF_TEMPLATE_DEFAULT}/config.json" && -f "${ORNITH_ROOT}/Ornith-1.0-35B-hf/config.json" ]]; then
  HF_TEMPLATE_DEFAULT="${ORNITH_ROOT}/Ornith-1.0-35B-hf"
fi
VENV="${ORNITH_ROOT}/.venv-public-py312"
PY="${PY:-${VENV}/bin/python}"
export PYTHONPATH="${ORNITH_ROOT}/sglang-fork/python:${KT_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"