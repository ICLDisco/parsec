/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_GPU_DATA_H_HAS_BEEN_INCLUDED
#define PARSEC_GPU_DATA_H_HAS_BEEN_INCLUDED

#include "parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/class/parsec_object.h"
#include "parsec/class/fifo.h"
#include "parsec/devices/device.h"

#if defined(PARSEC_HAVE_CUDA)
#include "parsec/class/list_item.h"
#include "parsec/class/list.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "parsec/utils/zone_malloc.h"

BEGIN_C_DECLS

#define PARSEC_GPU_USE_PRIORITIES     1

#define PARSEC_MAX_STREAMS            6
#define PARSEC_MAX_EVENTS_PER_STREAM  4
#define PARSEC_GPU_MAX_WORKSPACE      2
#define PARSEC_GPU_W2R_NB_MOVE_OUT    1

#if defined(PARSEC_PROF_TRACE)
#define PARSEC_PROFILE_CUDA_TRACK_DATA_IN  0x0001
#define PARSEC_PROFILE_CUDA_TRACK_DATA_OUT 0x0002
#define PARSEC_PROFILE_CUDA_TRACK_OWN      0x0004
#define PARSEC_PROFILE_CUDA_TRACK_EXEC     0x0008

extern int parsec_cuda_trackable_events;
extern int parsec_cuda_movein_key_start;
extern int parsec_cuda_movein_key_end;
extern int parsec_cuda_moveout_key_start;
extern int parsec_cuda_moveout_key_end;
extern int parsec_cuda_own_GPU_key_start;
extern int parsec_cuda_own_GPU_key_end;
#endif  /* defined(PROFILING) */

#define GPU_TASK_TYPE_D2HTRANSFER 111

extern float *device_load, *device_weight;

typedef struct __parsec_gpu_workspace {
    void* workspace[PARSEC_GPU_MAX_WORKSPACE];
    int stack_head;
    int total_workspace;
} parsec_gpu_workspace_t;

struct __parsec_gpu_context;
typedef struct __parsec_gpu_context parsec_gpu_context_t;

struct __parsec_gpu_exec_stream;
typedef struct __parsec_gpu_exec_stream parsec_gpu_exec_stream_t;

struct _gpu_device;
typedef struct _gpu_device gpu_device_t;

typedef int (*advance_task_function_t)(gpu_device_t            *gpu_device,
                                       parsec_gpu_context_t     *gpu_task,
                                       parsec_gpu_exec_stream_t *gpu_stream);

struct __parsec_gpu_context {
    parsec_list_item_t          list_item;
    parsec_execution_context_t *ec;
    advance_task_function_t    submit;
    int                        task_type;
    int                        pushout[MAX_PARAM_COUNT];
    const parsec_flow_t        *flow[MAX_PARAM_COUNT];
};

struct __parsec_gpu_exec_stream {
    struct __parsec_gpu_context **tasks;
    cudaEvent_t *events;
    cudaStream_t cuda_stream;
    int32_t max_events;  /* number of potential events, and tasks */
    int32_t executed;    /* number of executed tasks */
    int32_t start, end;  /* circular buffer management start and end positions */
    parsec_list_t *fifo_pending;
    parsec_gpu_workspace_t *workspace;
#if defined(PARSEC_PROF_TRACE)
    parsec_thread_profiling_t *profiling;
#endif  /* defined(PROFILING) */
#if defined(PARSEC_PROF_TRACE)
    int prof_event_track_enable;
    int prof_event_key_start, prof_event_key_end;
#endif  /* defined(PROFILING) */
};

struct _gpu_device {
    parsec_device_t super;
    uint8_t cuda_index;
    uint8_t major;
    uint8_t minor;
    uint8_t max_exec_streams;
    int16_t peer_access_mask;  /**< A bit set to 1 represent the capability of
                                *   the device to access directly the memory of
                                *   the index of the set bit device.
                                */
    parsec_gpu_exec_stream_t* exec_stream;
    volatile uint32_t mutex;
    parsec_list_t gpu_mem_lru;
    parsec_list_t gpu_mem_owned_lru;
    parsec_list_t pending;
    zone_malloc_t *memory;
    parsec_list_item_t *sort_starting_p;
};

#define PARSEC_CUDA_CHECK_ERROR( STR, ERROR, CODE )                      \
    do {                                                                \
        cudaError_t __cuda_error = (cudaError_t) (ERROR);               \
        if( cudaSuccess != __cuda_error ) {                             \
            parsec_warning( "%s:%d %s%s", __FILE__, __LINE__,          \
                    (STR), cudaGetErrorString(__cuda_error) );          \
            CODE;                                                       \
        }                                                               \
    } while(0)

int parsec_gpu_init(parsec_context_t *parsec_context);
int parsec_gpu_fini(void);

/**
 * Debugging functions.
 */
void dump_exec_stream(parsec_gpu_exec_stream_t* exec_stream);
void dump_GPU_state(gpu_device_t* gpu_device);

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

/**
 * Overload the default data_copy_t with a GPU specialized type
 */
typedef parsec_data_copy_t parsec_gpu_data_copy_t;

#include "parsec/data_distribution.h"

/* GPU workspace  ONLY works when PARSEC_ALLOC_GPU_PER_TILE is OFF */
int parsec_gpu_push_workspace(gpu_device_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream);
void* parsec_gpu_pop_workspace(gpu_device_t* gpu_device, parsec_gpu_exec_stream_t* gpu_stream, size_t size);
int parsec_gpu_free_workspace(gpu_device_t * gpu_device);

int parsec_gpu_get_best_device( parsec_execution_context_t* this_task, double ratio );

/* sort pending task list by number of spaces needed */
int parsec_gpu_sort_pending_list(gpu_device_t *gpu_device);
parsec_gpu_context_t* parsec_gpu_create_W2R_task(gpu_device_t *gpu_device, parsec_execution_unit_t *eu_context);
int parsec_gpu_W2R_task_fini(gpu_device_t *gpu_device, parsec_gpu_context_t *w2r_task, parsec_execution_unit_t *eu_context);

/**
 * Progress
 */
/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on CUDA events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_gpu_kernel_scheduler( parsec_execution_unit_t *eu_context,
                            parsec_gpu_context_t    *gpu_task,
                            int which_gpu );

/**
 * Predefined generic progress functions
 */

/**
 *  This function schedule the move of all the data required for a
 *  specific task from the main memory into the GPU memory.
 *
 *  Returns:
 *     a positive number: the number of data to be moved.
 *     -1: data cannot be moved into the GPU.
 *     -2: No more room on the GPU to move this data.
 */
int
parsec_gpu_kernel_push( gpu_device_t            *gpu_device,
                       parsec_gpu_context_t     *gpu_task,
                       parsec_gpu_exec_stream_t *gpu_stream);

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
int
parsec_gpu_kernel_pop( gpu_device_t            *gpu_device,
                      parsec_gpu_context_t     *gpu_task,
                      parsec_gpu_exec_stream_t *gpu_stream);

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
int
parsec_gpu_kernel_epilog( gpu_device_t        *gpu_device,
                         parsec_gpu_context_t *gpu_task );

int
parsec_gpu_kernel_cleanout( gpu_device_t        *gpu_device,
                            parsec_gpu_context_t *gpu_task );

END_C_DECLS

#endif /* defined(PARSEC_HAVE_CUDA) */

#endif
