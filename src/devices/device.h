/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_DEVICE_H_HAS_BEEN_INCLUDED
#define DAGUE_DEVICE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "list_item.h"
#if defined(DAGUE_PROF_TRACE)
#include "profiling.h"
#endif  /* defined(DAGUE_PROF_TRACE) */
#include "dague.h"

#define DAGUE_DEV_NONE       0x00
#define DAGUE_DEV_CPU        0x01
#define DAGUE_DEV_CUDA       0x02
#define DAGUE_DEV_INTEL_PHI  0x03
#define DAGUE_DEV_OPENCL     0x04
#define DAGUE_DEV_MAX        0x05

typedef struct dague_device_s dague_device_t;

typedef int (*dague_device_init_f)(dague_device_t*);
typedef int (*dague_device_fini_f)(dague_device_t*);
typedef int (*dague_device_handle_register_f)(dague_device_t*, dague_handle_t*);
typedef int (*dague_device_handle_unregister_f)(dague_device_t*, dague_handle_t*);
typedef int (*dague_device_memory_register_f)(dague_device_t*, void*, size_t);
typedef int (*dague_device_memory_unregister_f)(dague_device_t*, void*);

struct dague_device_s {
    dague_list_item_t item;

    /* Device Management Functions */
    dague_device_fini_f              device_fini;
    dague_device_handle_register_f   device_handle_register;
    dague_device_handle_unregister_f device_handle_unregister;
    dague_device_memory_register_f   device_memory_register;
    dague_device_memory_unregister_f device_memory_unregister;

    struct dague_context_s* context;  /**< The DAGuE context this device belongs too */
    char* name;  /**< Simple identified for the device */
    uint64_t transferred_data_in;
    uint64_t transferred_data_out;
    uint64_t required_data_in;
    uint64_t required_data_out;
    uint64_t executed_tasks;
    float device_sweight;  /**< Number of single precision operations per second */
    float device_dweight;  /**< Number of double precision operations per second */
#if defined(DAGUE_PROF_TRACE)
    dague_thread_profiling_t *profiling;
#endif  /* defined(PROFILING) */
    uint8_t device_index;
    uint8_t type;
};

extern uint32_t dague_nb_devices;
extern uint32_t dague_devices_mutex;
/**
 * Temporary variables used for load-balancing purposes.
 */
extern float *dague_device_load;
extern float *dague_device_sweight;
extern float *dague_device_dweight;

/**
 * Initialize the internal structures for managing external devices such as
 * accelerators and GPU. Memory nodes can as well be managed using the same
 * mechnism.
 */
extern int dague_devices_init(dague_context_t*);

/**
 * The runtime will shutdown, all internal structures have to be destroyed.
 */
extern int dague_devices_fini(dague_context_t*);

/**
 * Parse the list of potential devices and see which one would succesfully load
 * and initialize in the current environment.
 */
extern int dague_devices_select(dague_context_t*);

/**
 * This call mark the end of the configuration step, no devices can be registered
 * after this point. This gives a reference point regarding the number of available
 * (supported) devices.
 */
extern int dague_devices_freeze(dague_context_t*);
/**
 * Return a positive value if the devices configurations has been freezed by a call
 * to dague_devices_freeze().
 */
extern int dague_devices_freezed(dague_context_t*);

/**
 * Declare a new device with the runtime. The device will later provide a list
 * of supported operations.
 */
DAGUE_DECLSPEC int dague_devices_add(dague_context_t*, dague_device_t*);

/**
 * Retrieve a pointer to the registered device using the provided index.
 */
DAGUE_DECLSPEC dague_device_t* dague_devices_get(uint32_t);

/**
 * Remove the device from the list of enabled devices. All data residing on the
 * device will be copied back on the main memory (or the memory declared as the
 * originator of the data), and all tasks owned by the device will be discarded
 * and moved back into the main scheduling mechanism.
 */
DAGUE_DECLSPEC int dague_device_remove(dague_device_t* device);
/**
 * Returns the number of devices currently registered with the runtime. This
 * number can change until dague_devices_freeze() is called, fact that mark the
 * end of the configuration stage.
 */
static inline int dague_devices_enabled(void)
{
    return dague_nb_devices;
}

/**
 * Helper function for pretty-print. Compute the adapted unit,
 * annotated in human readable format ("K", "M", "G")
 */
void dague_compute_best_unit( uint64_t length,
                              float* updated_value,
                              char** best_unit );


#endif  /* DAGUE_DEVICE_H_HAS_BEEN_INCLUDED */
