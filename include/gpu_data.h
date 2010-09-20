/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED
#define DAGUE_GPU_DATA_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

#if defined(DAGUE_CUDA_SUPPORT)
#include "linked_list.h"
#include "dequeue.h"
#include "profiling.h"
#include "lifo.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#define DAGUE_SMART_SCHEDULING 0
#define DAGUE_MAX_STREAMS 4

typedef struct _gpu_device {
    dague_list_item_t item;
    CUcontext ctx;
    CUmodule hcuModule;
    CUfunction hcuFunction;
    CUstream streams[DAGUE_MAX_STREAMS];
    int max_streams;
    int id;
    int executed_tasks;
    int major;
    int minor;
    volatile uint32_t mutex;
#if DAGUE_SMART_SCHEDULING
    int lifoid;
#endif
    dague_dequeue_t pending;
    uint64_t transferred_data_in;
    uint64_t transferred_data_out;
    uint64_t required_data_in;
    uint64_t required_data_out;
    dague_linked_list_t* gpu_mem_lru;
#if defined(DAGUE_PROFILING)
    dague_thread_profiling_t *profiling;
#endif  /* defined(PROFILING) */
} gpu_device_t;

#define DAGUE_CUDA_CHECK_ERROR( STR, ERROR, CODE )                      \
    {                                                                   \
        cudaError_t __cuda_error = (cudaError_t) (ERROR);               \
        if( cudaSuccess != __cuda_error ) {                             \
            printf( "%s:%d %s%s\n", __FILE__, __LINE__,                 \
                    (STR), cudaGetErrorString(__cuda_error) );          \
            CODE;                                                       \
        }                                                               \
    }

extern gpu_device_t** gpu_devices;
int dague_gpu_init(int* puse_gpu, int dague_show_detailed_capabilities);
    
#endif /* defined(DAGUE_CUDA_SUPPORT) */

#endif
