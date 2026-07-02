# shellcheck shell=bash
# Source from ornith/*.sh — sets ORNITH_ROOT, PY, PYTHONPATH, KT_PYTHON.
_ornith_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORNITH_ROOT="${ORNITH_ROOT:-$(cd -- "${_ornith_script_dir}/../../../.." && pwd)}"
KT_ROOT="$(cd -- "${_ornith_script_dir}/../.." && pwd)"
VENV="${ORNITH_ROOT}/.venv-public-py312"
PY="${PY:-${VENV}/bin/python}"
export PYTHONPATH="${ORNITH_ROOT}/sglang-fork/python:${KT_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"