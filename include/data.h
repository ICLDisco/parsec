/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DATA_H_HAS_BEEN_INCLUDED
#define DATA_H_HAS_BEEN_INCLUDED

#include "dague_internal.h"
#include "dague/types.h"
#include <dague/sys/atomic.h>

/**
 * This is a variable changed only once, and contains the total number of
 * devices allowed to keep copies of a data. It is updated during the
 * initialization of the system and never changed after (!)
 */
extern uint32_t dague_supported_number_of_devices;

typedef uint8_t dague_data_coherency_t;
#define    DATA_COHERENCY_INVALID   ((dague_data_coherency_t)0x0)
#define    DATA_COHERENCY_OWNED     ((dague_data_coherency_t)0x1)
#define    DATA_COHERENCY_EXCLUSIVE ((dague_data_coherency_t)0x2)
#define    DATA_COHERENCY_SHARED    ((dague_data_coherency_t)0x4)

/**
 * This structure is the keeper of all the information regarding
 * each unique data that can be handled by the system. It contains
 * pointers to the versions managed by each supported devices.
 */
struct dague_data_s {
    uint32_t                  version;
    dague_data_coherency_t    coherency_state;
    uint16_t                  owner_device;
    dague_data_key_t          key;
    uint32_t                  nb_elts;          /* number of elements of the memory layout */
    struct dague_data_copy_s* device_copies[1]; /* this array allocated according to the number of devices
                                                (dague_supported_number_of_devices). It points to the most recent
                                                version of the data. */
};

typedef uint8_t dague_data_flag_t;
#define DAGUE_DATA_FLAG_ARENA     ((dague_data_flag_t)0x01)

/**
 * This structure represent a device copy of a dague_data_t.
 */
struct dague_data_copy_s {
    volatile uint32_t        refcount;

    uint8_t                  device_index;
    dague_data_flag_t        flags;
    dague_data_coherency_t   coherency_state;
    /* int8_t */

    int32_t                  readers;

    uint32_t                 version;

    struct _dague_data_copy* older;
    struct _dague_data*      original;
    void*                    device_private;
};

/**
 * Increase the refcount of this copy of the data.
 */
static inline uint32_t dague_data_copy_retain(dague_data_copy_t* data)
{
    return dague_atomic_inc_32b(&data->refcount);
}
#define DAGUE_DATA_COPY_RETAIN(DATA) dague_data_copy_retain(DATA)

/**
 * Decrease the refcount of this copy of the data. If the refcount reach
 * 0 the upper level is in charge of cleaning up and releasing all content
 * of the copy.
 */
static inline uint32_t dague_data_copy_release(dague_data_copy_t* data)
{
    return dague_atomic_dec_32b(&data->refcount);
    /* TODO: Move the copy back to the CPU before destroying it */
}
#define DAGUE_DATA_COPY_RELEASE(DATA) dague_data_copy_release(DATA)

/**
 * Return the device private pointer for a datacopy.
 */
static inline void* dague_data_copy_get_ptr(dague_data_copy_t* data)
{
    return data->device_private;
}
#define DAGUE_DATA_COPY_GET_PTR(DATA) dague_data_copy_get_ptr(DATA)

#endif  /* DATA_H_HAS_BEEN_INCLUDED */
