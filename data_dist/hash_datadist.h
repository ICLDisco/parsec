#ifndef HASH_DATATIST_H
#define HASH_DATATIST_H

#include "dague_config.h"

#include <stdarg.h>
#include <assert.h>

#include "data_distribution.h"
#include "data.h"
#include "vpmap.h"

typedef struct dague_hash_datadist_entry_s {
    uint32_t      key;  /**< Unique key */
    dague_data_t *data; /**< pointer to data meta information (if allocated) */
    /* User's parameters */
    void         *actual_data;
    int           rank;
    int           vpid;
    uint32_t      size;
    struct dague_hash_datadist_entry_s *next; /**< Next entry with the same hash */
} dague_hash_datadist_entry_t;

typedef struct dague_hash_datadist_s {
    dague_ddesc_t super;
    uint32_t hash_size;
    dague_hash_datadist_entry_t **hash;
} dague_hash_datadist_t;

/**
 * @FILE Interface for a hash-based PaRSEC data distribution.
 *
 * Usage:
 *  - Create the hash-based structure with dague_hash_datadist_create
 *  - Add each data element one after the other using dague_hash_datadist_set_data
 *    Each MPI rank must add each key at least with the rank. 
 *    data pointer and vpid must be defined only for the local node.
 *  - PaRSEC uses the data distribution
 *  - Destroy the structure with dague_hash_datadist_destroy
 */

/**
 * @PARAM [IN] np: the number of MPI ranks on which that data is distributed
 * @PARAM [IN] myrank: the rank of the calling process
 *
 * @RETURN the newly hash datadist (empty)
 */
dague_hash_datadist_t *dague_hash_datadist_create(int np, int myrank);

/**
 * @PARAM [IN] d: the datadist to destroy
 */
void dague_hash_datadist_destroy(dague_hash_datadist_t *d);

/**
 * @PARAM [INOUT] d: hash datadist on which we are adding a new data element
 * @PARAM [IN] actual_data: pointer to the memory area that hold the data
 *                          actual_data is NULL iff rank != myrank
 * @PARAM [IN] key: unique key to find the data (if the JDF writes A(x), x is the key)
 * @PARAM [IN] vpid: the vpid that hosts this data (undefined iff rank != myrank)
 * @PARAM [IN] rank: the rank that hosts this data
 * @PARAM [IN] size: the size in bytes of this data element
 */
void dague_hash_datadist_set_data(dague_hash_datadist_t *d, void *actual_data, uint32_t key, int vpid, int rank, uint32_t size);

#endif /* HASH_DATATIST_H */
