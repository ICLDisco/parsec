/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_OPENMP_DATA_H_HAS_BEEN_INCLUDED
#define PARSEC_OPENMP_DATA_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_internal.h"
#include "parsec/class/parsec_object.h"
#include "parsec/class/fifo.h"
#include "parsec/devices/device.h"

#if defined(PARSEC_HAVE_OPENMP)
#include "parsec/class/list_item.h"
#include "parsec/class/list.h"

#include <omp.h>

#include "parsec/utils/zone_malloc.h"

BEGIN_C_DECLS

#define PARSEC_OPENMP_MAX_WORKSPACE      2
#define PARSEC_OPENMP_W2R_NB_MOVE_OUT    1

typedef struct __parsec_openmp_workspace {
    void* workspace[PARSEC_OPENMP_MAX_WORKSPACE];
    int stack_head;
    int total_workspace;
} parsec_openmp_workspace_t;

#if defined(PARSEC_PROF_TRACE)
#define PARSEC_PROFILE_OMP_TRACK_DATA_IN  0x0001
#define PARSEC_PROFILE_OMP_TRACK_DATA_OUT 0x0002
#define PARSEC_PROFILE_OMP_TRACK_OWN      0x0004
#define PARSEC_PROFILE_OMP_TRACK_EXEC     0x0008

extern int parsec_omp_trackable_events;
extern int parsec_omp_movein_key_start;
extern int parsec_omp_movein_key_end;
extern int parsec_omp_moveout_key_start;
extern int parsec_omp_moveout_key_end;
extern int parsec_omp_own_GPU_key_start;
extern int parsec_omp_own_GPU_key_end;
#endif  /* defined(PROFILING) */

#define GPU_TASK_TYPE_D2HTRANSFER 111

struct __parsec_openmp_context;
typedef struct __parsec_openmp_context parsec_openmp_context_t;

struct __parsec_openmp_exec_stream;
typedef struct __parsec_openmp_exec_stream parsec_openmp_exec_stream_t;

struct _parsec_openmp_gpu_device;
typedef struct _parsec_openmp_gpu_device parsec_openmp_device_t;

typedef int (*advance_openmp_task_function_t)(parsec_openmp_device_t            *parsec_openmp_gpu_device,
                                       parsec_openmp_context_t     *gpu_task,
                                       parsec_openmp_exec_stream_t *gpu_stream);

struct __parsec_openmp_context {
    parsec_list_item_t          list_item;
    parsec_task_t *ec;
    advance_openmp_task_function_t    submit;
    int                        task_type;
    int                        pushout[MAX_PARAM_COUNT];
    const parsec_flow_t        *flow[MAX_PARAM_COUNT];
};

struct __parsec_openmp_exec_stream {
    struct __parsec_openmp_context **tasks;
    int *events;
    int omp_stream;
    int32_t max_events;  /* number of potential events, and tasks */
    int32_t executed;    /* number of executed tasks */
    int32_t start, end;  /* circular buffer management start and end positions */
    parsec_list_t *fifo_pending;
    parsec_openmp_workspace_t *workspace;
#if defined(PARSEC_PROF_TRACE)
    parsec_thread_profiling_t *profiling;
#endif  /* defined(PROFILING) */
#if defined(PARSEC_PROF_TRACE)
    int prof_event_track_enable;
    int prof_event_key_start, prof_event_key_end;
#endif  /* defined(PROFILING) */
};

struct _parsec_openmp_gpu_device {
    parsec_device_t super;
    uint8_t omp_index;
    uint8_t major;
    uint8_t minor;
    uint8_t max_exec_streams;
    int16_t peer_access_mask;  /**< A bit set to 1 represent the capability of
                                *   the device to access directly the memory of
                                *   the index of the set bit device.
                                */
    parsec_openmp_exec_stream_t* exec_stream;
    volatile uint32_t mutex;
    parsec_list_t gpu_mem_lru;
    parsec_list_t gpu_mem_owned_lru;
    parsec_list_t pending;
    zone_malloc_t *memory;
    parsec_list_item_t *sort_starting_p;
};


int parsec_omp_init(parsec_context_t *parsec_context);
int parsec_omp_fini(void);

/**
 * Debugging functions.
 */
void dump_omp_gpu_stream(parsec_openmp_exec_stream_t* exec_stream);
void dump_omp_state(parsec_openmp_device_t* parsec_openmp_gpu_device);

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/

/**
 * Overload the default data_copy_t with a GPU specialized type
 */
typedef parsec_data_copy_t parsec_openmp_data_copy_t;

#include "parsec/data_distribution.h"

/* GPU workspace  ONLY works when PARSEC_ALLOC_GPU_PER_TILE is OFF */
int parsec_openmp_push_workspace(parsec_openmp_device_t* parsec_openmp_gpu_device, parsec_openmp_exec_stream_t* gpu_stream);
void* parsec_openmp_pop_workspace(parsec_openmp_device_t* parsec_openmp_gpu_device, parsec_openmp_exec_stream_t* gpu_stream, size_t size);
int parsec_openmp_free_workspace(parsec_openmp_device_t * parsec_openmp_gpu_device);


/**
 * Progress
 */
/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for tranfers from the GPU into
 * the main memory. The synchronization on each stream is based on omp events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_openmp_kernel_scheduler(parsec_execution_stream_t *es,
                             parsec_openmp_context_t    *gpu_task,
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
parsec_openmp_kernel_push(parsec_openmp_device_t          *parsec_openmp_gpu_device,
                       parsec_openmp_context_t     *gpu_task,
                       parsec_openmp_exec_stream_t *gpu_stream);

/**
 *  This function schedule the move of all the modified data for a
 *  specific task from the GPU memory into the main memory.
 *
 *  Returns: negative number if any error occured.
 *           positive: the number of data to be moved.
 */
int
parsec_openmp_kernel_pop(parsec_openmp_device_t          *parsec_openmp_gpu_device,
                      parsec_openmp_context_t     *gpu_task,
                      parsec_openmp_exec_stream_t *gpu_stream);

/**
 * Make sure all data on the device is correctly put back into the queues.
 */
int
parsec_openmp_kernel_epilog(parsec_openmp_device_t      *parsec_openmp_gpu_device,
                         parsec_openmp_context_t *gpu_task );

int
parsec_openmp_kernel_cleanout(parsec_openmp_device_t       *parsec_openmp_gpu_device,
                            parsec_openmp_context_t *gpu_task );

END_C_DECLS

#endif /* defined(PARSEC_HAVE_OPENMP) */

#endif
