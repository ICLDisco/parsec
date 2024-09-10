/*
 * Copyright (c) 2019-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/sys/atomic.h"

#include "parsec/utils/mca_param.h"
#include "parsec/utils/debug.h"
#include "parsec/constants.h"
#include "parsec/mca/device/template/device_template.h"

static int
parsec_template_memory_register(parsec_device_module_t* device,
                                parsec_data_collection_t* desc,
                                void* ptr, size_t length)
{
    int rc = PARSEC_ERROR;

    /* Memory needs to be registered only once with the device. One registration = one deregistration */
    if (desc->memory_registration_status == MEMORY_STATUS_UNREGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_UNREGISTERED;

    (void)device; (void)ptr; (void)length;
    return rc;
}

static int
parsec_template_memory_unregister(parsec_device_module_t* device,
                                  parsec_data_collection_t* desc,
                                  void* ptr)
{
    int rc = PARSEC_ERROR;

    /* Memory needs to be registered only once with the device. One registration = one deregistration */
    if (desc->memory_registration_status == MEMORY_STATUS_UNREGISTERED) {
        rc = PARSEC_SUCCESS;
        return rc;
    }

    rc = PARSEC_SUCCESS;
    desc->memory_registration_status = MEMORY_STATUS_UNREGISTERED;

    (void)device; (void)ptr;
    return rc;
}

/**
 * Register a taskpool with a device by checking that the device
 * supports the dynamic function required by the different incarnations.
 * If multiple devices of the same type exists we assume thay all have
 * the same capabilities.
 */
static int
parsec_template_taskpool_register(parsec_device_module_t* device,
                                  parsec_taskpool_t* tp)
{
    parsec_device_template_module_t* dev = (parsec_device_template_module_t*)device;
    const char* paths[] = { ".", NULL};
    int32_t rc = PARSEC_ERR_NOT_FOUND;
    uint32_t i, j;

    /**
     * Detect if a particular chore has a dynamic load dependency and if yes
     * load the corresponding module and find the function.
     */
    assert(tp->devices_index_mask & (1 << device->device_index));

    for( i = 0; i < tp->nb_task_classes; i++ ) {
        const parsec_task_class_t* tc = tp->task_classes_array[i];
        __parsec_chore_t* chores = (__parsec_chore_t*)tc->incarnations;
        for( j = 0; NULL != chores[j].hook; j++ ) {
            if( chores[j].type & device->type )
                continue;
            if( NULL != chores[j].dyld_fn ) {
                /* the function has been set for another device of the same type */
                return PARSEC_SUCCESS;
            }
            if ( NULL == chores[j].dyld ) {
                chores[j].dyld_fn = NULL;  /* No dynamic support required for this kernel */
                rc = PARSEC_SUCCESS;
            } else {
                void* devf = parsec_device_find_function(chores[j].dyld, ".", paths);
                if( NULL != devf ) {
                    chores[j].dyld_fn = devf;
                    rc = PARSEC_SUCCESS;
                }
            }
        }
    }
    if( PARSEC_SUCCESS != rc ) {
        tp->devices_index_mask &= ~(1 << device->device_index);  /* drop support for this device */
        parsec_debug_verbose(10, parsec_template_output_stream,
                             "Device %d (%s) disabled for taskpool %p", device->device_index, device->name, tp);
    }
    (void)dev;
    return rc;
}

static int
parsec_template_taskpool_unregister(parsec_device_module_t* device, parsec_taskpool_t* tp)
{
    (void)device; (void)tp;
    return PARSEC_SUCCESS;
}

/**
 * Attach a device to a PaRSEC context. A device can only be attached to
 * a single context at the time.
 */
static int
parsec_device_template_attach( parsec_device_template_module_t* device,
                               parsec_context_t* context )
{
    return parsec_mca_device_add(context, (parsec_device_module_t*)device);
}

/**
 * Detach a device from a context. Both the context and the device remain
 * valid, they are simply disconnected.
 * This function should only be called once all tasks and all data related to the
 * context has been removed from the device.
 */
static int
parsec_device_template_detach( parsec_device_template_module_t* device,
                               parsec_context_t* context )
{
    (void)context;
    return parsec_mca_device_remove((parsec_device_module_t*)device);
}

int
parsec_device_template_module_init( int deviceid, parsec_device_module_t** module )
{
    parsec_device_template_module_t* device;
    int show_caps_index, show_caps = 0;

    show_caps_index = parsec_mca_param_find("device", NULL, "show_capabilities"); 
    if(0 < show_caps_index) {
        parsec_mca_param_lookup_int(show_caps_index, &show_caps);
    }

    *module = NULL;

    device = (parsec_device_template_module_t*)calloc(1, sizeof(parsec_device_template_module_t));
    PARSEC_OBJ_CONSTRUCT(device, parsec_device_template_module_t);
    device->super.name = strdup("0");

    device->super.type                 = PARSEC_DEV_TEMPLATE;
    device->super.executed_tasks       = 0;
    device->super.transferred_data_in  = 0;
    device->super.transferred_data_out = 0;
    device->super.required_data_in     = 0;
    device->super.required_data_out    = 0;
    device->super.nb_evictions         = 0;

    device->super.attach              = (parsec_device_attach_f)parsec_device_template_attach;
    device->super.detach              = (parsec_device_detach_f)parsec_device_template_detach;
    device->super.memory_register     = parsec_template_memory_register;
    device->super.memory_unregister   = parsec_template_memory_unregister;
    device->super.taskpool_register   = parsec_template_taskpool_register;
    device->super.taskpool_unregister = parsec_template_taskpool_unregister;

    device->super.gflops_fp16 = 0;  /* no computational capacity */
    device->super.gflops_tf32 = 0;
    device->super.gflops_fp32 = 0;
    device->super.gflops_fp64 = 0;

    if( show_caps ) {
        parsec_inform("TEMPLATE Device %d enabled\n", device->super.device_index);
    }

    (void)deviceid;
    *module = (parsec_device_module_t*)device;
    return PARSEC_SUCCESS;
}

int
parsec_device_template_module_fini(parsec_device_module_t* device)
{
    parsec_device_template_module_t* dev = (parsec_device_template_module_t*)device;

    /* mostly nothing to do for the template device */
    (void)dev; (void)device;
    return PARSEC_SUCCESS;
}

