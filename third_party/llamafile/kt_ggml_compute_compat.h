// kt_ggml_compute_compat.h — bridging ik_llama.cpp ggml for vendored llamafile
//
// ik_llama.cpp removed enum ggml_task_type and the `type` field from
// struct ggml_compute_params (it lives only in ggml.c without type).
// The vendored llamafile code (Mozilla-Ocho 0.8.8) expects the original
// ggerganov/llama.cpp API where ggml_compute_params had .type and
// GGML_TASK_TYPE_{INIT,COMPUTE,FINALIZE} were defined.
//
// This header provides those definitions so the vendored llamafile
// compiles against ik_llama.cpp headers.  It must be included AFTER
// "llama.cpp/ggml-impl.h" (which includes ggml.h).
//
// WARNING: the struct layout defined here differs from ik's internal
// ggml_compute_params (defined in ggml.c).  Do NOT pass a struct
// allocated with this definition into ik's ggml_graph_compute() —
// ik will read wrong offsets.  The vendored llamafile functions that
// accept const ggml_compute_params* (llamafile_mixmul et al.) are
// compiled but never called from ik code, so the mismatch is benign.
//
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#pragma once

#include <stddef.h>  // size_t
#include "llama.cpp/ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_task_type {
    GGML_TASK_TYPE_INIT     = 0,
    GGML_TASK_TYPE_COMPUTE  = 1,
    GGML_TASK_TYPE_FINALIZE = 2,
};

// Opaque forward decl — ik defines this in ggml.c, we only need a pointer.
struct ggml_compute_state_shared;

struct ggml_compute_params {
    enum ggml_task_type type;

    int ith;
    int nth;

    size_t wsize;
    void * wdata;

    struct ggml_compute_state_shared * shared;
};

#ifdef __cplusplus
}

extern "C" {
    ggml_type_traits_t ggml_internal_get_type_traits(ggml_type type);
}

inline ggml_type_traits_t kt_ggml_internal_get_type_traits_wrapper(ggml_type type) {
    ggml_type_traits_t traits = (ggml_internal_get_type_traits)(type);
    if (traits.vec_dot_type == 341 || traits.vec_dot_type == 340 ||
        traits.vec_dot_type == 99 || traits.vec_dot_type == 98 || traits.vec_dot_type == 97) {
        traits.vec_dot_type = GGML_TYPE_Q8_K;
    }
    return traits;
}

#define ggml_internal_get_type_traits kt_ggml_internal_get_type_traits_wrapper
#endif
