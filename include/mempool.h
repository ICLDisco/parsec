/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _mempool_h
#define _mempool_h

#include "dague_config.h"
#include "lifo.h"

typedef struct dague_mempool_s dague_mempool_t;
typedef struct dague_thread_mempool_s dague_thread_mempool_t;

/**
 * each element that is allocated from a mempool must
 * keep somewhere a pointer to the thread_mempool that allocated it
 * This is needed to be pushed back to the correct memory pool.
 *
 * When creating a memory pool for a specific type, the size of
 * the type, and the offset to this pointer in the type, must
 * be passed to the main memory pool creation, as well as the
 * number of threads. Then, the thread-specific memory pool
 * must be constructed by each of the threads.
 *
 * Memory Pool memory must also be a dague_list_item_t, to
 * be chained using Lifos.
 */

struct dague_mempool_s {
    unsigned int            nb_thread_mempools;
    size_t                  elt_size;
    size_t                  pool_owner_offset;  /**< this is the offset to get to the thread_mempool_t 
                                                 *   from a newly allocated element */
    volatile uint32_t       nb_max_elt;         /**< this reflects the maximum of the nb_elt of the other threads */
    dague_thread_mempool_t *thread_mempools;
};

struct dague_thread_mempool_s {
    dague_mempool_t  *parent;    /**<  back pointer to the mempool */
    uint32_t nb_elt;             /**< this is the number of elements this thread
                                  *   has allocated since the creation of the pool */
    dague_lifo_t mempool;
};

/** DAGUE_MEMPOOL_CONSTRUCT / dague_mempool_construct
 *    One can use either of the interfaces.
 *    DAGUE_MEMPOOL_CONSTRUCT( &mempool, dague_execution_context_t, mempool, nbcores );
 *  has the same effect as
 *    dague_mempool_construct( &mempool, sizeof(dague_execution_context_t), 
 *                             (char*)&context.mempool - (char*)&context, nbcores );
 *  The macro is provided as a simplification to compute the offset of the mempool field in
 *  the type of elements that is allocated by this memory pool.
 * 
 *  Once the system-wide memory pool has been constructed, each thread
 *  can take its onw mempool->thread_mempools element.
 */
#define DAGUE_MEMPOOL_CONSTRUCT( mempool, type, field_name, nbthreads ) \
    do {                                                                \
        type __pseudo_elt;                                              \
        dague_mempool_construct( (mempool), sizeof(type),               \
                                 (char*)&(__pseudo_elt.##field_name) -  \
                                 (char*)&(__pseudo_elt),                \
                                 nbthreads );                           \
        } while(0)
void dague_mempool_construct( dague_mempool_t *mempool,
                              size_t elt_size,
                              size_t pool_offset,
                              unsigned int nbthreads );

/** dague_thread_mempool_allocate_when_empty
 *    Internal function.
 *    allocates an element of size thread_mempool->parent->elt_size,
 *    and set the back pointe to the appropriate thread_mempool,
 *    when the requested thread_mempool is empty.
 *  This function is called by dague_thread_mempool_allocate,
 *  and should never be called by another function.
 */
void *dague_thread_mempool_allocate_when_empty( dague_thread_mempool_t *thread_mempool );

/** dague_thread_mempool_allocate 
 *    allocates an element of size thread_mempool->mempool->elt_size,
 *    using the internal function if the pool is empty.
 */
static inline void *dague_thread_mempool_allocate( dague_thread_mempool_t *thread_mempool )
{
    unsigned char *ret;
    ret = (unsigned char *)dague_lifo_pop( &thread_mempool->mempool );
    if( ret == NULL ) {
        ret = dague_thread_mempool_allocate_when_empty( thread_mempool );
    }
    return ret;
}

/** dague_mempool_free
 *     "frees" an element allocated by one of the thread_mempools,
 *     pushing it back to it's owner memory pool
 */
static inline void  dague_mempool_free( dague_mempool_t *mempool, void *elt )
{
    unsigned char *_elt = (unsigned char *)elt;
    dague_thread_mempool_t *owner = *(dague_thread_mempool_t **)(_elt + mempool->pool_owner_offset);
    dague_lifo_push( &(owner->mempool), elt );
}

/** dague_thread_mempool_free
 *     a shortcut to dague_mempool_free( thread_mempool->parent, elt );
 */
static inline void  dague_thread_mempool_free( dague_thread_mempool_t *thread_mempool, void *elt )
{
    dague_mempool_free( thread_mempool->parent, elt );
}

/** dague_mempool_destruct
 *    destroy all resources allocated with the system-wide memory pool
 *    and the thread-specific memory pools. Anything that has not been
 *    pushed back in one of the thread-specific memory pools before
 *    might be lost.
 */
void dague_mempool_destruct( dague_mempool_t *mempool );

#endif /* defined(_mempool_h) */
