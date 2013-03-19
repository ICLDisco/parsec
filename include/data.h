/*
 * Copyright (c) 2012-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DATA_H_HAS_BEEN_INCLUDED
#define DATA_H_HAS_BEEN_INCLUDED

#include "dague_internal.h"
#include <dague/types.h>
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
    dague_object_t            super;

    uint32_t                  version;
    dague_data_coherency_t    coherency_state;
    int8_t                    owner_device;
    dague_data_key_t          key;
    uint32_t                  nb_elts;          /* number of elements of the memory layout */
    struct dague_data_copy_s *device_copies[1]; /* this array allocated according to the number of devices
                                                 * (dague_supported_number_of_devices). It points to the most recent
                                                 * version of the data.
                                                 */
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_data_t);

typedef uint8_t dague_data_flag_t;
#define DAGUE_DATA_FLAG_ARENA     ((dague_data_flag_t)0x01)

/**
 * This structure represent a device copy of a dague_data_t.
 */
struct dague_data_copy_s {
    dague_list_item_t         super;

    int8_t                    device_index;         /**< Index in the original->device_copies array */
    dague_data_flag_t         flags;
    dague_data_coherency_t    coherency_state;
    /* int8_t */

    int32_t                   readers;

    uint32_t                  version;

    struct dague_data_copy_s *older;                 /**< unused yet */
    dague_data_t             *original;
    void*                    device_private;         /**< The pointer to the device-specific data.
                                                      *   Overlay data distributions assume that arithmetic
                                                      *   can be done on these pointers.
                                                      */
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_data_copy_t);

/**
 * Initialize the DAGuE data infrastructure
 */
DAGUE_DECLSPEC int dague_data_init(dague_context_t* context);
DAGUE_DECLSPEC int dague_data_fini(dague_context_t* context);

static inline dague_data_copy_t*
dague_data_get_copy(dague_data_t* data, uint32_t device)
{
    return data->device_copies[device];
}

dague_data_copy_t* dague_data_copy_new(dague_data_t* data, uint8_t device);
/**
 * Decrease the refcount of this copy of the data. If the refcount reach
 * 0 the upper level is in charge of cleaning up and releasing all content
 * of the copy.
 */
static inline void dague_data_copy_release(dague_data_copy_t* data)
{
    /* TODO: Move the copy back to the CPU before destroying it */
    OBJ_RELEASE(data);
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

/**
 * Allocate a new data structure set to INVALID and no attached copies.
 */
extern dague_data_t* dague_data_new(void);

/**
 * Force the release of a data (which become unavailable for further uses).
 */
extern void dague_data_delete(dague_data_t* data);

/**
 * Attach a new copy corresponding to the specified device to a data. If a copy
 * for the device is already attached, nothing will be done and an error code
 * will be returned.
 */
extern int dague_data_copy_attach(dague_data_t* data,
                                  dague_data_copy_t* copy,
                                  uint8_t device);
extern int dague_data_copy_detach(dague_data_t* data,
                                  dague_data_copy_t* copy,
                                  uint8_t device);

int dague_data_transfer_ownership_to_copy(dague_data_t* map,
                                          uint8_t device,
                                          uint8_t access_mode);
extern void dague_dump_data_copy(dague_data_copy_t* copy);
extern void dague_dump_data(dague_data_t* copy);

#endif  /* DATA_H_HAS_BEEN_INCLUDED */
