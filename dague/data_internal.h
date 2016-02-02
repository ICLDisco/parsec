/*
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DATA_INTERNAL_H_HAS_BEEN_INCLUDED
#define DATA_INTERNAL_H_HAS_BEEN_INCLUDED

#include "dague/dague_internal.h"
#include "dague/data.h"
#include "dague/types.h"

/**
 * This is a variable changed only once, and contains the total number of
 * devices allowed to keep copies of a data. It is updated during the
 * initialization of the system and never changed after (!)
 */
extern uint32_t dague_supported_number_of_devices;

/**
 * This structure is the keeper of all the information regarding
 * each unique data that can be handled by the system. It contains
 * pointers to the versions managed by each supported devices.
 */
struct dague_data_s {
    dague_object_t            super;

    dague_data_key_t          key;
    int8_t                    owner_device;
    struct dague_ddesc_s*     ddesc;
    uint32_t                  nb_elts;          /* number of elements of the memory layout */
    struct dague_data_copy_s *device_copies[1]; /* this array allocated according to the number of devices
                                                 * (dague_supported_number_of_devices). It points to the most recent
                                                 * version of the data.
                                                 */
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_data_t);

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
    dague_arena_chunk_t      *arena_chunk;           /**< If this is an arena-based data, keep
                                                      *   the chunk pointer here, to avoid
                                                      *   risky pointers arithmetic (pointers mis-alignment
                                                      *   depending on many parameters) */
    void                     *device_private;        /**< The pointer to the device-specific data.
                                                      *   Overlay data distributions assume that arithmetic
                                                      *   can be done on these pointers. */
    dague_data_status_t      data_transfer_status;   /** three status */
    struct dague_execution_context_s *push_task;            /** the task who actually do the PUSH */
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_data_copy_t);

#define DAGUE_DATA_GET_COPY(DATA, DEVID) \
    ((DATA)->device_copies[(DEVID)])
/**
 * Decrease the refcount of this copy of the data. If the refcount reach
 * 0 the upper level is in charge of cleaning up and releasing all content
 * of the copy.
 */
#define DAGUE_DATA_COPY_RELEASE(DATA)     \
    do {                                  \
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Release data copy %p at %s:%d\n", (DATA), __FILE__, __LINE__); \
        OBJ_RELEASE((DATA));                                            \
    } while(0)

/**
 * Return the device private pointer for a datacopy.
 */
#define DAGUE_DATA_COPY_GET_PTR(DATA) \
    ((DATA)->device_private)

#endif  /* DATA_INTERNAL_H_HAS_BEEN_INCLUDED */
