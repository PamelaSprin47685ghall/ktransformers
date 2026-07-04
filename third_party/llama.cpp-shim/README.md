# llama.cpp include shim

This directory explains the `third_party/llama.cpp/` include shim.

## Purpose

`kt-kernel` C++ sources include ggml headers via `#include "llama.cpp/ggml-impl.h"` etc.
Previously `third_party/llama.cpp` was the ggerganov/llama.cpp submodule with a flat layout
where every header lived at the repo root.

After migrating to `ik_llama.cpp` (PamelaSprin47685ghall/ik_llama.cpp), ggml headers
live in a structured layout:

| kt-kernel include                      | ik_llama.cpp source              |
|----------------------------------------|----------------------------------|
| `llama.cpp/ggml.h`                     | `ggml/include/ggml.h`            |
| `llama.cpp/ggml-impl.h`                | `ggml/src/ggml-impl.h`           |
| `llama.cpp/ggml-quants.h`              | `ggml/src/ggml-quants.h`         |
| `llama.cpp/ggml-common.h`              | `ggml/src/ggml-common.h`         |
| `llama.cpp/ggml-alloc.h`               | `ggml/include/ggml-alloc.h`      |
| `llama.cpp/ggml-backend.h`             | `ggml/include/ggml-backend.h`    |
| `llama.cpp/ggml-backend-impl.h`        | `ggml/src/ggml-backend-impl.h`   |

The shim directory `third_party/llama.cpp/` is a **symlink farm** – each file in the
flat namespace is a relative symlink into the corresponding ik file. This lets
existing `#include "llama.cpp/..."` code compile without changes.

## Workflow

After submodule update / fresh clone:

```bash
cd ktransformers
bash scripts/install-llama-cpp-include-shim.sh
```

The script fails early if `third_party/ik_llama.cpp/` is missing.

## How it fits

```
ktransformers/
  third_party/
    ik_llama.cpp/        ← submodule (PamelaSprin47685ghall/ik_llama.cpp)
    llama.cpp/           ← shim  (symlinks; not a git submodule)
      ggml.h            → ../ik_llama.cpp/ggml/include/ggml.h
      ggml-impl.h       → ../ik_llama.cpp/ggml/src/ggml-impl.h
      ...
    llama.cpp-shim/      ← this README
  scripts/
    install-llama-cpp-include-shim.sh
  kt-kernel/CMakeLists.txt  → add_subdirectory(../third_party/ik_llama.cpp)
```

`kt-kernel/CMakeLists.txt` builds `ik_llama.cpp` and links its `llama` target.
`include_directories(../third_party)` makes `llama.cpp/ggml-impl.h` resolvable.

## Vendored llamafile compat

`third_party/llamafile/` is a vendored copy of Mozilla-Ocho/llamafile 0.8.8 written
against the **original** ggerganov/llama.cpp API.  ik_llama.cpp removed two things
that this vendored code depends on:

1. `enum ggml_task_type` and the `type` field in `struct ggml_compute_params`
   (ik's ggml.c defines `ggml_compute_params` without `.type` internally).
2. The types `GGML_TYPE_Q8_0_X4` / `GGML_TYPE_Q8_1_X4` are **already** in
   ik's `ggml_type` enum (values 97, 98); the vendored `iqk_mul_mat.inc`
   redefined them as `constexpr` with different values (98, 99), causing
   a conflict.

The shim `kt_ggml_compute_compat.h` (in the llamafile directory) provides:

- `enum ggml_task_type` → `GGML_TASK_TYPE_INIT`, `COMPUTE`, `FINALIZE`
- `struct ggml_compute_params` with a `.type` field (layout differs from ik's
  internal struct — see the header warning)
- Forward declaration of `struct ggml_compute_state_shared`

`tinyblas_cpu.h` includes this header after the ggml includes.
`iqk_mul_mat.inc` no longer defines the constexpr duplicates.

**Caveat**: The `ggml_compute_params` layout defined here does **not** match
ik's internal struct.  Functions that accept `const ggml_compute_params*`
(e.g. `llamafile_mixmul`) are compiled but never called from ik code, so the
mismatch is benign.  A full llamafile sync from ik's own
`ggml/src/llamafile/sgemm.h` (simpler signature, no `ggml_compute_params`) is
the proper long-term fix.
