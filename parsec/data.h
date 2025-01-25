/*
 * Copyright (c) 2012-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DATA_H_HAS_BEEN_INCLUDED
#define PARSEC_DATA_H_HAS_BEEN_INCLUDED

#include "parsec/runtime.h"
#include "parsec/datatype.h"

/** @defgroup parsec_internal_data Data
 *  @ingroup parsec_internal
 *    Data objects represent the meta-information associated to each
 *    user's or temporary data blocks that the PaRSEC runtime engine
 *    manipulate.
 *  @addtogroup parsec_internal_data
 *  @{
 */

BEGIN_C_DECLS

typedef uint64_t parsec_data_key_t;
struct parsec_context_s;

typedef uint8_t parsec_data_coherency_t;
#define    PARSEC_DATA_COHERENCY_INVALID   ((parsec_data_coherency_t)0x0)
#define    PARSEC_DATA_COHERENCY_OWNED     ((parsec_data_coherency_t)0x1)
#define    PARSEC_DATA_COHERENCY_EXCLUSIVE ((parsec_data_coherency_t)0x2)
#define    PARSEC_DATA_COHERENCY_SHARED    ((parsec_data_coherency_t)0x4)

typedef uint8_t parsec_data_status_t;
#define    PARSEC_DATA_STATUS_NOT_TRANSFER          ((parsec_data_coherency_t)0x0)
#define    PARSEC_DATA_STATUS_UNDER_TRANSFER        ((parsec_data_coherency_t)0x1)
#define    PARSEC_DATA_STATUS_COMPLETE_TRANSFER     ((parsec_data_coherency_t)0x2)
/**
 * Data copies have three levels of 'ownership':
 * - a data copy can be owned and managed by PaRSEC.
 *   This is typically the data copies coming from arenas
 *   and most gpu data copies. PaRSEC knows how to move them
 *   allocate and free them. They have the flag
 *   PARSEC_DATA_FLAG_PARSEC_OWNED | PARSEC_DATA_FLAG_PARSEC_MANAGED
 * - it can be managed but not owned by PaRSEC. This is typically
 *   the case of data collections: PaRSEC knows how to create
 *   duplicates of this copy, but cannot release its device_private,
 *   and is not responsible of the life cycle of the device_private
 *   pointer. Such copies have the flag PARSEC_DATA_FLAG_PARSEC_MANAGED
 * - it can be neither owned nor managed by PaRSEC. This is typically
 *   the case of user-managed copies on the GPU. The user is responsible
 *   for the lifecycle of the data copy, and trying to move such data
 *   copy (create a duplicate on the same or another device) should
 *   not happen and if needed will issue a runtime error
 **/
typedef uint8_t parsec_data_flag_t;
#define PARSEC_DATA_FLAG_ARENA          ((parsec_data_flag_t)1<<0)
#define PARSEC_DATA_FLAG_TRANSIT        ((parsec_data_flag_t)1<<1)
#define PARSEC_DATA_FLAG_PARSEC_MANAGED ((parsec_data_flag_t)1<<6)
#define PARSEC_DATA_FLAG_PARSEC_OWNED   ((parsec_data_flag_t)1<<7)

/**
 * Initialize the PaRSEC data infrastructure
 */
PARSEC_DECLSPEC int parsec_data_init(struct parsec_context_s* context);
PARSEC_DECLSPEC int parsec_data_fini(struct parsec_context_s* context);

PARSEC_DECLSPEC parsec_data_copy_t*
parsec_data_get_copy(parsec_data_t* data, uint32_t device);

PARSEC_DECLSPEC parsec_data_copy_t*
parsec_data_copy_new(parsec_data_t* data, uint8_t device,
                     parsec_datatype_t dtt, parsec_data_flag_t flags);

/**
 * Decrease the refcount of this copy of the data. If the refcount reach
 * 0 the upper level is in charge of cleaning up and releasing all content
 * of the copy.
 */
PARSEC_DECLSPEC void parsec_data_copy_release(parsec_data_copy_t* copy);

/**
 * Return the device private pointer for a datacopy.
 */
PARSEC_DECLSPEC void* parsec_data_copy_get_ptr(parsec_data_copy_t* data);

/**
 * Return the device private pointer for a data on a specified device.
 */
PARSEC_DECLSPEC void* parsec_data_get_ptr(parsec_data_t* data, uint32_t device);

/**
 * Allocate a new data structure set to INVALID and no attached copies.
 */
PARSEC_DECLSPEC parsec_data_t* parsec_data_new(void);

/**
 * Force the release of a data (which become unavailable for further uses).
 */
PARSEC_DECLSPEC void parsec_data_delete(parsec_data_t* data);

/**
 * Attach a new copy corresponding to the specified device to a data. If a copy
 * for the device is already attached, nothing will be done and an error code
 * will be returned.
 */
PARSEC_DECLSPEC int
parsec_data_copy_attach(parsec_data_t* data,
                       parsec_data_copy_t* copy,
                       uint8_t device);
PARSEC_DECLSPEC int
parsec_data_copy_detach(parsec_data_t* data,
                        parsec_data_copy_t* copy,
                        uint8_t device);

PARSEC_DECLSPEC int
parsec_data_transfer_ownership_to_copy(parsec_data_t* map,
                                      uint8_t device,
                                      uint8_t access_mode);
PARSEC_DECLSPEC int
parsec_data_start_transfer_ownership_to_copy(parsec_data_t* data,
                                             uint8_t device,
                                             uint8_t access_mode);
PARSEC_DECLSPEC void
parsec_data_end_transfer_ownership_to_copy(parsec_data_t* data,
                                                uint8_t device,
                                                uint8_t access_mode);
PARSEC_DECLSPEC void parsec_dump_data_copy(parsec_data_copy_t* copy);
PARSEC_DECLSPEC void parsec_dump_data(parsec_data_t* copy);

PARSEC_DECLSPEC parsec_data_t *
parsec_data_create( parsec_data_t **holder,
                   parsec_data_collection_t *desc,
                   parsec_data_key_t key,
                   void *ptr, size_t size, parsec_data_flag_t flags);

parsec_data_t*
parsec_data_create_with_type( parsec_data_collection_t *desc,
                              parsec_data_key_t key, void *ptr, size_t size,
                              parsec_datatype_t dtt);

/**
 * Destroy the parsec_data_t generated through a call to parsec_data_create
 */
PARSEC_DECLSPEC void
parsec_data_destroy( parsec_data_t *holder );

END_C_DECLS

/** @} */

#endif  /* PARSEC_DATA_H_HAS_BEEN_INCLUDED */
