/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DATA_H_HAS_BEEN_INCLUDED
#define DATA_H_HAS_BEEN_INCLUDED

#include "dague_internal.h"

/**
 * This is a variable changed only once, and contains the total number of
 * devices allowed to keep copies of a data. It is updated during the
 * initialization of the system and never changed after (!)
 */
uint32_t dague_supported_number_of_devices;

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
struct _dague_data {
    dague_arena_t*           arena;
    uint32_t                 nb_elts;          /* number of elements of the memory layout */
    dague_data_key_t         key;
    dague_data_coherency_t   coherency_state;
    uint16_t                 owner_device;
    uint32_t                 version;
    struct _dague_data_copy* device_copies[1]; /* this array allocated according to the number of devices
                                                (dague_supported_number_of_devices). It points to the most recent
                                                version of the data. */
};

typedef uint8_t dague_data_flag_t;
#define DAGUE_DATA_FLAG_ARENA     ((dague_data_flag_t)0x01)

/**
 * This structure represent a device copy of a dague_data_t.
 */
struct _dague_data_copy {
    volatile uint32_t        refcount;
    struct _dague_data_copy* older;
    struct _dague_data*      original;
    dague_data_coherency_t   coherency_state;
    uint8_t                  device_index;
    dague_data_flag_t        flags;
    int16_t                  readers;
    uint32_t                 version;
    void*                    device_private;
};

static inline uint32_t dague_data_copy_ref(dague_data_copy_t* data)
{
    return dague_atomic_inc_32b(&data->refcount);
}

/**
 * Decrease the refcount of this copy of the data. If the refcount reach
 * 0 the upper level is in charge of cleaning up and releasing all content
 * of the copy.
 */
static inline uint32_t dague_data_copy_unref(dague_data_copy_t* data)
{
    return dague_atomic_dec_32b(&data->refcount);
}

static inline dague_data_copy_t* dague_data_copy_detach(dague_data_copy_t* data)
{
    dague_data_t* original = data->original;
    dague_data_copy_t* recent = original->device_copies[data->device_index];

    assert( 0 == data->readers );
    assert( data->version == data->original->version );
    assert( NULL == data->device_private );

    /* remove this copy from the chain */
    if( recent == data ) {
        original->device_copies[data->device_index] = data->older;
    } else {
        while( recent->older != data ) recent = recent->older;
        recent->older = data->older;
        data->older = NULL;
    }
    return original->device_copies[data->device_index];
}

#endif  /* DATA_H_HAS_BEEN_INCLUDED */
