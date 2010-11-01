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

#define DAGUE_MAX_STREAMS 4
#define DAGUE_MAX_EVENTS_PER_STREAM  4

#if defined(DAGUE_PROFILING)
extern int dague_cuda_movein_key_start;
extern int dague_cuda_movein_key_end;
extern int dague_cuda_moveout_key_start;
extern int dague_cuda_moveout_key_end;
#endif  /* defined(PROFILING) */

typedef struct _gpu_device {
    dague_list_item_t item;
    CUcontext ctx;
    CUmodule hcuModule;
    CUfunction hcuFunction;
    CUstream streams[DAGUE_MAX_STREAMS];
    int max_streams;
#if !defined(DAGUE_GPU_STREAM_PER_TASK)
    int max_in_tasks,
        max_exec_tasks,
        max_out_tasks;
    int max_exec_streams;
    struct dague_execution_context_t **in_array;
    struct dague_execution_context_t **exec_array;
    struct dague_execution_context_t **out_array;
    CUevent *in_array_events;
    CUevent *exec_array_events;
    CUevent *out_array_events;
    int in_submit, in_waiting,
        exec_submit, exec_waiting,
        out_submit, out_waiting;
    struct dague_fifo_t *fifo_pending_in;
    struct dague_fifo_t *fifo_pending_exec;
    struct dague_fifo_t *fifo_pending_out;
#endif  /* DAGUE_GPU_STREAM_PER_TASK */
    int id;
    int executed_tasks;
    int major;
    int minor;
    volatile uint32_t mutex;
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
    
/**
 * Enable GPU-compatible memory if possible
 */
void dague_data_enable_gpu( int nbgpu );

/**
 * returns not false iff dague_data_enable_gpu succeeded
 */
int dague_using_gpu(void);

/**
 * allocate a buffer to hold the data using GPU-compatible memory if needed
 */
void* dague_allocate_data( size_t matrix_size );

/**
 * free a buffer allocated by dague_allocate_data
 */
void dague_free_data(void *address);

#endif /* defined(DAGUE_CUDA_SUPPORT) */

#endif
