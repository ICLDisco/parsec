/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "mempool.h"
#ifdef PARSEC_HAVE_STRING_H
#include <string.h>
#endif

/** parsec_thread_mempool_construct
 *    constructs the thread-specific memory pool.
 */
static void parsec_thread_mempool_construct( parsec_thread_mempool_t *thread_mempool,
                                            parsec_mempool_t *mempool )
{
    thread_mempool->parent = mempool;
    OBJ_CONSTRUCT(&thread_mempool->mempool, parsec_lifo_t);
    thread_mempool->nb_elt = 0;
}

static void parsec_thread_mempool_destruct( parsec_thread_mempool_t *thread_mempool )
{
    void *elt;
    while(NULL != (elt = parsec_lifo_pop(&thread_mempool->mempool))) {
        if(NULL != thread_mempool->parent->obj_class) {
            parsec_lifo_item_free(elt);
        } else {
            free(elt);
        }
    }
    OBJ_DESTRUCT(&thread_mempool->mempool);
}

void parsec_mempool_construct( parsec_mempool_t *mempool,
                              parsec_class_t* obj_class, size_t elt_size,
                              size_t pool_offset,
                              unsigned int nbthreads )
{
    uint32_t tid;

    mempool->nb_thread_mempools = nbthreads;
    mempool->elt_size = elt_size < sizeof(parsec_list_item_t) ? sizeof(parsec_list_item_t) : elt_size;
    mempool->pool_owner_offset = pool_offset;
    mempool->nb_max_elt = 0;
    mempool->obj_class = obj_class;
    mempool->thread_mempools = (parsec_thread_mempool_t *)malloc(sizeof(parsec_thread_mempool_t) * nbthreads);
    memset( mempool->thread_mempools, 0, sizeof(parsec_thread_mempool_t) * nbthreads );

    for(tid = 0; tid < mempool->nb_thread_mempools; tid++)
        parsec_thread_mempool_construct(&mempool->thread_mempools[tid], mempool);
}

uint64_t parsec_mempool_destruct( parsec_mempool_t *mempool )
{
    uint32_t tid;
    uint64_t usage_counter = 0;

    for(tid = 0; tid < mempool->nb_thread_mempools; tid++) {
        usage_counter += mempool->thread_mempools[tid].nb_elt;
        parsec_thread_mempool_destruct(&mempool->thread_mempools[tid]);
    }

    free(mempool->thread_mempools);
    mempool->thread_mempools = NULL;
    mempool->nb_thread_mempools = 0;

    return usage_counter;
}

void *parsec_thread_mempool_allocate_when_empty( parsec_thread_mempool_t *thread_mempool )
{
    /*
     * Simple heuristic: don't try to balance things between threads,
     * just allocate within this thread
     */
    void *elt;
    parsec_thread_mempool_t **owner;

    elt = parsec_lifo_item_alloc(&thread_mempool->mempool, thread_mempool->parent->elt_size );
    owner = (parsec_thread_mempool_t **)((char*)elt + thread_mempool->parent->pool_owner_offset);
    *owner = thread_mempool;
    if( NULL != thread_mempool->parent->obj_class ) {
        OBJ_CONSTRUCT_INTERNAL(elt, thread_mempool->parent->obj_class);
    }
    thread_mempool->nb_elt++;
    return elt;
}
