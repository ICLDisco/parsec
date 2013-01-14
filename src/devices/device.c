/*
 *
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <dague/devices/device.h>
#include <dague/constants.h>

#include <stdlib.h>

#include "gpu_data.h"

uint32_t dague_nb_devices = 0;
static uint32_t dague_nb_max_devices = 0;
static uint32_t dague_devices_freezed = 0;
uint32_t dague_devices_mutex = 0;  /* unlocked */
dague_device_t** dague_devices = NULL;

int dague_devices_init(void)
{
    return DAGUE_ERR_NOT_IMPLEMENTED;
}

int dague_devices_fini(void)
{
    return dague_gpu_fini();
}

int dague_devices_freeze(void)
{
    if(dague_devices_freezed)
        return -1;
    dague_devices_freezed = 1;
    return 0;
}

int dague_devices_select(dague_context_t* dague_context)
{
    return dague_gpu_init(dague_context);
}

int dague_devices_add(dague_device_t* device)
{
    if( dague_devices_freezed ) {
        return -1;
    }
    dague_atomic_lock(&dague_devices_mutex);
    if( (dague_nb_devices+1) > dague_nb_max_devices ) {
        if( NULL == dague_devices ) /* first time */
            dague_nb_max_devices = 4;
        else
            dague_nb_max_devices *= 2; /* every other time */
        dague_devices = realloc(dague_devices, dague_nb_max_devices * sizeof(dague_device_t*));
    }
    dague_devices[dague_nb_devices] = device;
    device->device_index = dague_nb_devices;
    dague_nb_devices++;
    return device->device_index;
}

dague_device_t* dague_devices_get(uint32_t device_index)
{
    if( device_index > dague_nb_devices )
        return NULL;
    return dague_devices[device_index];
}

int dague_device_remove(dague_device_t* device)
{
    dague_atomic_lock(&dague_devices_mutex);
    if(NULL == dague_devices[device->device_index])
        return -1;
    dague_devices[device->device_index] = NULL;
    device->device_index = -1;
    dague_atomic_unlock(&dague_devices_mutex);
    return 0;
}

int dague_devices_enabled(void)
{
    return dague_nb_devices;
}

