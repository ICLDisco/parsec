/*
 * Copyright (c) 2010-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED

#include "parsec/mca/device/cuda/device_cuda.h"

#if defined(PARSEC_HAVE_CUDA)

BEGIN_C_DECLS

#define PARSEC_MAX_STREAMS            6
#define PARSEC_MAX_EVENTS_PER_STREAM  4
#define PARSEC_GPU_MAX_WORKSPACE      2

#if defined(PARSEC_PROF_TRACE)
#define PARSEC_PROFILE_CUDA_TRACK_DATA_IN  0x0001
#define PARSEC_PROFILE_CUDA_TRACK_DATA_OUT 0x0002
#define PARSEC_PROFILE_CUDA_TRACK_OWN      0x0004
#define PARSEC_PROFILE_CUDA_TRACK_EXEC     0x0008
#define PARSEC_PROFILE_CUDA_TRACK_MEM_USE  0x0010
#define PARSEC_PROFILE_CUDA_TRACK_PREFETCH 0x0020

extern int parsec_cuda_trackable_events;
extern int parsec_cuda_movein_key_start;
extern int parsec_cuda_movein_key_end;
extern int parsec_cuda_moveout_key_start;
extern int parsec_cuda_moveout_key_end;
extern int parsec_cuda_own_GPU_key_start;
extern int parsec_cuda_own_GPU_key_end;
extern int parsec_cuda_allocate_memory_key;
extern int parsec_cuda_free_memory_key;
extern int parsec_cuda_use_memory_key_start;
extern int parsec_cuda_use_memory_key_end;
extern int parsec_cuda_prefetch_key_start;
extern int parsec_cuda_prefetch_key_end;
extern int parsec_device_cuda_one_profiling_stream_per_cuda_stream;
#endif  /* defined(PROFILING) */

#define GPU_TASK_TYPE_KERNEL       0x0000
#define GPU_TASK_TYPE_D2HTRANSFER  0x1000
#define GPU_TASK_TYPE_PREFETCH     0x2000
#define GPU_TASK_TYPE_WARMUP       0x4000
#define GPU_TASK_TYPE_D2D_COMPLETE 0x8000

/* From MCA parameters */
extern int use_cuda_index, use_cuda;
extern int cuda_mask, cuda_verbosity;
extern int cuda_memory_block_size, cuda_memory_percentage, cuda_memory_number_of_blocks;
extern char* cuda_lib_path;
extern int32_t parsec_CUDA_sort_pending_list;

PARSEC_DECLSPEC extern const parsec_device_module_t parsec_device_cuda_module;

typedef struct parsec_gpu_workspace_s {
    void* workspace[PARSEC_GPU_MAX_WORKSPACE];
    int stack_head;
    int total_workspace;
} parsec_gpu_workspace_t;


/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

int parsec_cuda_module_init( int device, parsec_device_module_t** module );
int parsec_cuda_module_fini(parsec_device_module_t* device);

END_C_DECLS

#if defined(PARSEC_PROF_TRACE)
typedef struct {
    uint64_t size;
    uint64_t data_key;
    uint64_t dc_id;
} parsec_device_cuda_memory_prof_info_t;
#define PARSEC_DEVICE_CUDA_MEMORY_PROF_INFO_CONVERTER "size{int64_t};data_key{uint64_t};dc_id{uint64_t}"
#endif /* PARSEC_PROF_TRACE */

#endif /* defined(PARSEC_HAVE_CUDA) */

#endif  /* PARSEC_DEVICE_CUDA_INTERNAL_H_HAS_BEEN_INCLUDED */
