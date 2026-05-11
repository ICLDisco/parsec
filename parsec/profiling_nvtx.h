/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#ifndef PARSEC_PROFILING_NVTX_H_HAS_BEEN_INCLUDED
#define PARSEC_PROFILING_NVTX_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"

#if defined(PARSEC_PROF_TRACE_NVTX)

#include <stdint.h>

#include "parsec/profiling.h"

typedef struct parsec_profiling_nvtx_range_s parsec_profiling_nvtx_range_t;

int parsec_profiling_nvtx_register_mca(void);
int parsec_profiling_nvtx_is_enabled(void);
void parsec_profiling_nvtx_init(int process_id);
void parsec_profiling_nvtx_fini(void);
void parsec_profiling_nvtx_register_key(int key, const char *name,
                                        const char *attributes);
void parsec_profiling_nvtx_dictionary_flush(void);
void parsec_profiling_nvtx_release_stream(parsec_profiling_nvtx_range_t **active,
                                          parsec_profiling_nvtx_range_t **freelist);
void parsec_profiling_nvtx_trace(parsec_profiling_nvtx_range_t **active,
                                 parsec_profiling_nvtx_range_t **freelist,
                                 int key, int is_start,
                                 uint64_t event_id, uint32_t taskpool_id);

#endif /* defined(PARSEC_PROF_TRACE_NVTX) */

#endif /* PARSEC_PROFILING_NVTX_H_HAS_BEEN_INCLUDED */
