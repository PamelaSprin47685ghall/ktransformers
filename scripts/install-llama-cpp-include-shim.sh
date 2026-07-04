#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# install-llama-cpp-include-shim.sh
#
# Creates third_party/llama.cpp/ as a symlink farm pointing into
# third_party/ik_llama.cpp/.  After running this script,
#   #include "llama.cpp/ggml-impl.h"
# resolves through the flat shim to the structured ik file.
#
# Must be run from the ktransformers repo root.
# Fails early if third_party/ik_llama.cpp is missing.

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || realpath "$(dirname "$0")/..")"

IK_DIR="third_party/ik_llama.cpp"
SHIM_DIR="third_party/llama.cpp"
# Symlink targets are relative to ${SHIM_DIR}/ (not repo root).
IK_REL="../ik_llama.cpp"

if [ ! -d "${IK_DIR}/ggml" ]; then
    echo "ERROR: ${IK_DIR} not found or incomplete." >&2
    echo "Did you run 'git submodule update --init --recursive'?" >&2
    exit 1
fi

# Remove stale shim if it exists (must not be a git submodule)
if [ -d "${SHIM_DIR}" ] && [ -f "${SHIM_DIR}/.git" ]; then
    echo "ERROR: ${SHIM_DIR} appears to be a git submodule. Remove it first:" >&2
    echo "  git submodule deinit third_party/llama.cpp" >&2
    echo "  git rm third_party/llama.cpp" >&2
    exit 1
fi

mkdir -p "${SHIM_DIR}"

# Map: shim_filename -> ik_relative_path
declare -A LINKS
LINKS[ggml.h]="${IK_REL}/ggml/include/ggml.h"
LINKS[ggml-alloc.h]="${IK_REL}/ggml/include/ggml-alloc.h"
LINKS[ggml-backend.h]="${IK_REL}/ggml/include/ggml-backend.h"
LINKS[ggml-impl.h]="${IK_REL}/ggml/src/ggml-impl.h"
LINKS[ggml-quants.h]="${IK_REL}/ggml/src/ggml-quants.h"
LINKS[ggml-common.h]="${IK_REL}/ggml/src/ggml-common.h"
LINKS[ggml-backend-impl.h]="${IK_REL}/ggml/src/ggml-backend-impl.h"

for name in "${!LINKS[@]}"; do
    target="${LINKS[$name]}"
    link="${SHIM_DIR}/${name}"
    if [ -L "${link}" ] && [ "$(readlink "${link}")" = "${target}" ]; then
        continue  # already correct
    fi
    ln -sf "${target}" "${link}"
    echo "  ${link} -> ${target}"
done

echo "llama.cpp include shim installed at ${SHIM_DIR}/"
