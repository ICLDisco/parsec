/*
 * Copyright (c) 2023      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_LEVEL_ZERO_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_LEVEL_ZERO_H_HAS_BEEN_INCLUDED

#include "parsec.h"

#if defined(PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT)
#include "parsec/mca/device/device.h"
#include "parsec/mca/device/device_gpu.h"
#include "parsec/mca/device/level_zero/device_level_zero_dpcpp.h"

#include <level_zero/ze_api.h>

BEGIN_C_DECLS

struct parsec_level_zero_task_s;
typedef struct parsec_level_zero_task_s parsec_level_zero_task_t;

struct parsec_level_zero_exec_stream_s;
typedef struct parsec_level_zero_exec_stream_s parsec_level_zero_exec_stream_t;

struct parsec_device_level_zero_module_s;
typedef struct parsec_device_level_zero_module_s parsec_device_level_zero_module_t;

struct parsec_device_level_zero_driver_s;
typedef struct parsec_device_level_zero_driver_s parsec_device_level_zero_driver_t;

struct parsec_level_zero_workspace_s;
typedef struct parsec_level_zero_workspace_s parsec_level_zero_workspace_t;

extern parsec_device_base_component_t parsec_device_level_zero_component;

struct parsec_level_zero_task_s {
    parsec_gpu_task_t   super;
};

struct parsec_device_level_zero_driver_s {
    ze_driver_handle_t               ze_driver;
    ze_context_handle_t              ze_context;
    uint32_t                         ref_count;
    parsec_sycl_wrapper_platform_t  *swp;
};

struct parsec_device_level_zero_module_s {
    parsec_device_gpu_module_t         super;
    uint8_t                            level_zero_index;
    parsec_device_level_zero_driver_t *driver;
    ze_device_handle_t                 ze_device;
    parsec_sycl_wrapper_device_t      *swd;
    int                                memory_index;
};

PARSEC_OBJ_CLASS_DECLARATION(parsec_device_level_zero_module_t);

struct parsec_level_zero_exec_stream_s {
    parsec_gpu_exec_stream_t super;
    /* There is exactly one task per active fence (max_fences being the uppoer bound).
     * Upon fence completion the complete_stage function associated with the task is
     * called, and this will decide what is going on next with the task. If the task
     * remains in the system the function is supposed to update it.
     */
    ze_fence_handle_t               *fences;
    ze_command_list_handle_t        *command_lists;
    ze_command_queue_handle_t        level_zero_cq;
    parsec_sycl_wrapper_queue_t     *swq;
};

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/
/**
 * Overload the default data_copy_t with a GPU specialized type
 */
typedef parsec_data_copy_t parsec_gpu_data_copy_t;

END_C_DECLS

#define PARSEC_LEVEL_ZERO_CHECK_ERROR(STR, ERROR, CODE)                       \
    do {                                                                      \
        if( ZE_RESULT_SUCCESS != (ERROR) ) {                                  \
            parsec_warning( "%s:%d %s returns Error 0x%x", __FILE__, __LINE__,\
                            (STR), (ERROR) );                                 \
            CODE;                                                             \
        }                                                                     \
    } while(0)
#else
#error "should be defined"
#endif /* defined(PARSEC_HAVE_DEV_LEVEL_ZERO_SUPPORT) */

#endif  /* PARSEC_DEVICE_LEVEL_ZERO_H_HAS_BEEN_INCLUDED */
