/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "mempool.h"
#ifdef HAVE_STRING_H
#include <string.h>
#endif

/** dague_thread_mempool_construct
 *    constructs the thread-specific memory pool.
 */
static void dague_thread_mempool_construct( dague_thread_mempool_t *thread_mempool,
                                            dague_mempool_t *mempool )
{
    thread_mempool->parent = mempool;
    OBJ_CONSTRUCT(&thread_mempool->mempool, dague_lifo_t);
    thread_mempool->nb_elt = 0;
}

static void dague_thread_mempool_destruct( dague_thread_mempool_t *thread_mempool )
{
    void *elt;
    while(NULL != (elt = dague_lifo_pop(&thread_mempool->mempool)))
        DAGUE_LIFO_ITEM_FREE(elt);
    OBJ_DESTRUCT(&thread_mempool->mempool);
}

void dague_mempool_construct( dague_mempool_t *mempool,
                              dague_class_t* obj_class, size_t elt_size,
                              size_t pool_offset,
                              unsigned int nbthreads )
{
    uint32_t tid;

    mempool->nb_thread_mempools = nbthreads;
    mempool->elt_size = elt_size;
    mempool->pool_owner_offset = pool_offset;
    mempool->nb_max_elt = 0;
    mempool->obj_class = obj_class;
    mempool->thread_mempools = (dague_thread_mempool_t *)malloc(sizeof(dague_thread_mempool_t) * nbthreads);
    memset( mempool->thread_mempools, 0, sizeof(dague_thread_mempool_t) * nbthreads );

    for(tid = 0; tid < mempool->nb_thread_mempools; tid++)
        dague_thread_mempool_construct(&mempool->thread_mempools[tid], mempool);
}

void dague_mempool_destruct( dague_mempool_t *mempool )
{
    uint32_t tid;

    for(tid = 0; tid < mempool->nb_thread_mempools; tid++)
        dague_thread_mempool_destruct(&mempool->thread_mempools[tid]);

    free(mempool->thread_mempools);
    mempool->thread_mempools = NULL;
    mempool->nb_thread_mempools = 0;
}

void *dague_thread_mempool_allocate_when_empty( dague_thread_mempool_t *thread_mempool )
{
    /*
     * Simple heuristic: don't try to balance things between threads,
     * just allocate within this thread
     */
    void *elt;
    dague_thread_mempool_t **owner;

    DAGUE_LIFO_ITEM_ALLOC(&thread_mempool->mempool, elt, thread_mempool->parent->elt_size );
    owner = (dague_thread_mempool_t **)((char*)elt + thread_mempool->parent->pool_owner_offset);
    *owner = thread_mempool;
    if( NULL != thread_mempool->parent->obj_class ) {
        OBJ_CONSTRUCT_INTERNAL(elt, thread_mempool->parent->obj_class);
    }
    return elt;
}
