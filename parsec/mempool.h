/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _mempool_h
#define _mempool_h

#include "parsec_config.h"
#include "parsec/class/lifo.h"

/** @addtogroup parsec_internal_mempool
 *  @{
 */

typedef struct parsec_mempool_s parsec_mempool_t;
typedef struct parsec_thread_mempool_s parsec_thread_mempool_t;

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
 * Memory Pool memory must also be a parsec_list_item_t, to
 * be chained using LIFOs.
 */
struct parsec_mempool_s {
    unsigned int            nb_thread_mempools; /**< Number of thread mempools that share this mempool */
    size_t                  elt_size;           /**< Size of one element in this mempool */
    size_t                  pool_owner_offset;  /**< this is the offset to get to the thread_mempool_t
                                                 *   from a newly allocated element */
    volatile uint32_t       nb_max_elt;         /**< this reflects the maximum of the nb_elt of the other threads */
    parsec_class_t          *obj_class;         /**< the base class of the objects inside the mempool */
    parsec_thread_mempool_t *thread_mempools;   /**< Array of thread mempools (of size nb_thread_mempools) */
};

struct parsec_thread_mempool_s {
    parsec_mempool_t  *parent;   /**<  back pointer to the mempool */
    uint32_t nb_elt;             /**< this is the number of elements this thread
                                  *   has allocated since the creation of the pool */
    parsec_lifo_t mempool;       /**< Elements are stored in a LIFO */
};

/**
 * @brief constructs a mempool
 *
 * @details
 *    One can use either of the interfaces.
 *    PARSEC_MEMPOOL_CONSTRUCT( &mempool, parsec_execution_context_t, mempool, nbcores );
 *  has the same effect as
 *    parsec_mempool_construct( &mempool, sizeof(parsec_execution_context_t),
 *                             (char*)&context.mempool - (char*)&context, nbcores );
 *  The macro is provided as a simplification to compute the offset of the mempool field in
 *  the type of elements that is allocated by this memory pool.
 *
 *  Once the system-wide memory pool has been constructed, each thread
 *  can take its onw mempool->thread_mempools element.
 */
#define PARSEC_MEMPOOL_CONSTRUCT( mempool, type, field_name, nbthreads )    \
    do {                                                                   \
        type __pseudo_elt;                                                 \
        parsec_mempool_construct( (mempool), OBJ_CLASS(type), sizeof(type), \
                                 (char*)&(__pseudo_elt.##field_name) -     \
                                 (char*)&(__pseudo_elt),                   \
                                 nbthreads );                              \
        } while(0)
/**
 * @brief mempool constructor (outside of the parsec_object_t model)
 *
 * @details
 * Create a mempool for nbthreads, where each element of size elt_size
 * are ojects of class obj_class, and the field at byte pool_offset of
 * these elements is the backpointer to the mempool object for memory
 * release
 *
 * @param[out] mempool the mempool to construct
 * @param[in] ojb_class the class of objects stored in this mempool
 * @param[in] elt_size the size of each objects in this mempool
 * @param[in] pool_offset the number of bytes between the beginning of the object
 *            and a parsec_thread_mempool_t pointer field inside the object
 *            that points to the mempool that allocated the object
 * @param[in] the number of threads that can share this mempool
 */
void parsec_mempool_construct( parsec_mempool_t *mempool,
                              parsec_class_t* obj_class, size_t elt_size,
                              size_t pool_offset,
                              unsigned int nbthreads );

/**
 * @brief extends a thread-mempool when it is empty
 *
 * @details
 *    Internal function.
 *    allocates an element of size thread_mempool->parent->elt_size,
 *    and set the back pointe to the appropriate thread_mempool,
 *    when the requested thread_mempool is empty.
 *  This function is called by parsec_thread_mempool_allocate,
 *  and should never be called by another function.
 *
 * @param[inout] thread_mempool the thread mempool to extend
 * @return an object of class obj_class
 */
void *parsec_thread_mempool_allocate_when_empty( parsec_thread_mempool_t *thread_mempool );

/**
 * @brief allocate an element from a mempool
 *
 * @details
 *    allocates an element of size thread_mempool->mempool->elt_size,
 *    using the internal function if the pool is empty.
 *
 * @param[inout] thread_mempool the thread-mempool from which an element should be allocated
 * @return the new element
 */
static inline void *parsec_thread_mempool_allocate( parsec_thread_mempool_t *thread_mempool )
{
    void* ret;
    ret = (void*)parsec_lifo_pop( &thread_mempool->mempool );
    if( ret == NULL ) {
        ret = parsec_thread_mempool_allocate_when_empty( thread_mempool );
    }
    return ret;
}

/**
 * @brief return a mempool element to its mempool
 *
 * @details
 *     "frees" an element allocated by one of the thread_mempools,
 *     pushing it back to it's owner memory pool
 * @param[inout] mempool the mempool to which the element should be returned
 * @param[inout] elt the element that is pushed back to the mempool
 */
static inline void  parsec_mempool_free( parsec_mempool_t *mempool, void *elt )
{
    unsigned char *_elt = (unsigned char *)elt;
    parsec_thread_mempool_t *owner = *(parsec_thread_mempool_t **)(_elt + mempool->pool_owner_offset);
    if(NULL != owner)
        parsec_lifo_push( &(owner->mempool), (parsec_list_item_t*)elt );
}

/**
 * @brief return a mempool element to its thread-mempool
 *
 * @details
 *     a shortcut to parsec_mempool_free( thread_mempool->parent, elt );
 *
 * @param[inout] thread_mempool the thread-mempool to which elt should be returned
 * @param[inout] elt the element to free
 */
static inline void  parsec_thread_mempool_free( parsec_thread_mempool_t *thread_mempool, void *elt )
{
    parsec_mempool_free( thread_mempool->parent, elt );
}

/**
 * @brief destroy a mempool and collect all allocated memory
 *
 * @details
 *    destroy all resources allocated with the system-wide memory pool
 *    and the thread-specific memory pools. Anything that has not been
 *    pushed back in one of the thread-specific memory pools before
 *    might be lost.
 *
 * @param[inout] mempool the mempool to destruct
 */
void parsec_mempool_destruct( parsec_mempool_t *mempool );

/** @} */

#endif /* defined(_mempool_h) */
