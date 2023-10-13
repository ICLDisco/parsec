/*
 * Copyright (c) 2010-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_CUDA_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_CUDA_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_internal.h"
#include "parsec/class/parsec_object.h"
#include "parsec/mca/device/device.h"

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include "parsec/class/list_item.h"
#include "parsec/class/list.h"
#include "parsec/class/fifo.h"
#include "parsec/mca/device/device_gpu.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

BEGIN_C_DECLS

struct parsec_cuda_task_s;
typedef struct parsec_cuda_task_s parsec_cuda_task_t;

struct parsec_cuda_exec_stream_s;
typedef struct parsec_cuda_exec_stream_s parsec_cuda_exec_stream_t;

struct parsec_device_cuda_module_s;
typedef struct parsec_device_cuda_module_s parsec_device_cuda_module_t;

struct parsec_cuda_workspace_s;
typedef struct parsec_cuda_workspace_s parsec_cuda_workspace_t;

extern parsec_device_base_component_t parsec_device_cuda_component;

struct parsec_cuda_task_s {
    parsec_gpu_task_t   super;
};

struct parsec_device_cuda_module_s {
    parsec_device_gpu_module_t super;
    uint8_t                    cuda_index;
    uint8_t                    major;
    uint8_t                    minor;
};

PARSEC_OBJ_CLASS_DECLARATION(parsec_device_cuda_module_t);

struct parsec_cuda_exec_stream_s {
    parsec_gpu_exec_stream_t super;
    /* There is exactly one task per active event (max_events being the upper bound).
     * Upon event completion the complete_stage function associated with the task is
     * called, and this will decide what is going on next with the task. If the task
     * remains in the system the function is supposed to update it.
     */
    cudaEvent_t               *events;
    cudaStream_t               cuda_stream;
};

/**
 * Overload the default data_copy_t with a GPU specialized type
 */
typedef parsec_data_copy_t parsec_gpu_data_copy_t;

#define PARSEC_CUDA_CHECK_ERROR( STR, ERROR, CODE )                     \
    do {                                                                \
        cudaError_t __cuda_error = (cudaError_t) (ERROR);               \
        if( cudaSuccess != __cuda_error ) {                             \
            parsec_warning( "%s:%d %s%s", __FILE__, __LINE__,           \
                            (STR), cudaGetErrorString(__cuda_error) );  \
            CODE;                                                       \
        }                                                               \
    } while(0)

/**
* Progress
*/
/**
 * This version is based on 4 streams: one for transfers from the memory to
 * the GPU, 2 for kernel executions and one for transfers from the GPU into
 * the main memory. The synchronization on each stream is based on GPU events,
 * such an event indicate that a specific epoch of the lifetime of a task has
 * been completed. Each type of stream (in, exec and out) has a pending FIFO,
 * where tasks ready to jump to the respective step are waiting.
 */
parsec_hook_return_t
parsec_cuda_kernel_scheduler( parsec_execution_stream_t *es,
                              parsec_gpu_task_t    *gpu_task,
                              int which_gpu );

/* Default stage_in function to transfer data to the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_cuda_stage_in(parsec_gpu_task_t        *gtask,
                             uint32_t                  flow_mask,
                             parsec_gpu_exec_stream_t *gpu_stream);

/* Default stage_out function to transfer data from the GPU device.
 * Transfer transfer the <count> contiguous bytes from
 * task->data[i].data_in to task->data[i].data_out.
 *
 * @param[in] task parsec_task_t containing task->data[i].data_in, task->data[i].data_out.
 * @param[in] flow_mask indicating task flows for which to transfer.
 * @param[in] gpu_stream parsec_gpu_exec_stream_t used for the transfer.
 *
 */
int
parsec_default_cuda_stage_out(parsec_gpu_task_t        *gtask,
                              uint32_t                  flow_mask,
                              parsec_gpu_exec_stream_t *gpu_stream);

END_C_DECLS

#endif /* defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) */

#endif  /* PARSEC_DEVICE_CUDA_H_HAS_BEEN_INCLUDED */
