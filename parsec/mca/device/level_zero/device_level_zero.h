/*
 * Copyright (c) 2021      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_LEVEL_ZERO_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_LEVEL_ZERO_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_internal.h"
#include "parsec/class/parsec_object.h"
#include "parsec/mca/device/device.h"

#if defined(PARSEC_HAVE_LEVEL_ZERO)
#include "parsec/class/list_item.h"
#include "parsec/class/list.h"
#include "parsec/class/fifo.h"
#include "parsec/mca/device/device_gpu.h"

#include <ze_api.h>

BEGIN_C_DECLS

struct parsec_level_zero_task_s;
typedef struct parsec_level_zero_task_s parsec_level_zero_task_t;

struct parsec_level_zero_exec_stream_s;
typedef struct parsec_level_zero_exec_stream_s parsec_level_zero_exec_stream_t;

struct parsec_device_level_zero_module_s;
typedef struct parsec_device_level_zero_module_s parsec_device_level_zero_module_t;

struct parsec_level_zero_workspace_s;
typedef struct parsec_level_zero_workspace_s parsec_level_zero_workspace_t;

extern parsec_device_base_component_t parsec_device_level_zero_component;

struct parsec_level_zero_task_s {
    parsec_gpu_task_t   super;
};

struct parsec_device_level_zero_module_s {
    parsec_device_gpu_module_t       super;
    uint8_t                          level_zero_index;
    ze_driver_handle_t               ze_driver;
    ze_device_handle_t               ze_device;
    ze_context_handle_t              ze_context;
};

PARSEC_OBJ_CLASS_DECLARATION(parsec_device_level_zero_module_t);

struct parsec_level_zero_exec_stream_s {
    parsec_gpu_exec_stream_t super;
    /* There is exactly one task per active event (max_events being the uppoer bound).
     * Upon event completion the complete_stage function associated with the task is
     * called, and this will decide what is going on next with the task. If the task
     * remains in the system the function is supposed to update it.
     */
    ze_event_handle_t               *events;
    ze_event_pool_handle_t           ze_event_pool;
    ze_command_list_handle_t         level_zero_cl;
};

/****************************************************
 ** GPU-DATA Specific Starts Here **
 ****************************************************/
PARSEC_DECLSPEC extern int parsec_level_zero_output_stream;

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

#endif /* defined(PARSEC_HAVE_LEVEL_ZERO) */

#endif  /* PARSEC_DEVICE_LEVEL_ZERO_H_HAS_BEEN_INCLUDED */
