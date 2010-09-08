/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_ARENA_H__
#define __USE_ARENA_H__

#include <stdlib.h>

#include "dague_config.h"
#if defined(HAVE_STDDEF_H)
#include <stddef.h>
#endif  /* HAVE_STDDEF_H */
#include "debug.h"
#include "stats.h"

#include "atomic.h"
#include "lifo.h"

typedef struct dague_arena_t
{
    dague_atomic_lifo_t lifo;
    size_t alignment; /* alignment to be respected, elem_size should be >> alignment, prefix size is the minimum alignment */
    size_t elem_size; /* size of one element */
    volatile int32_t used; /* elements currently out of the arena */
    int32_t max_used; /* maximum size of the arena in elements */
    volatile int32_t released; /* elements currently not used but allocated */
    int32_t max_released; /* when more that max elements are released, they are really freed instead of joining the lifo */
    /* some host hardware requires special allocation functions (Cuda, pinning,
     * Open CL, ...). Defaults are to use C malloc/free */
    void* (*malloc)(size_t size);
    void (*free)(void* ptr);
} dague_arena_t;

typedef struct dague_arena_elem_prefix_t {
    dague_arena_t* origin;
    void* data;
    volatile uint32_t refcount;
    uint32_t cache_friendly_emptyness;
} dague_arena_elem_prefix_t;

/* types used to compute alignment  */
union _internal_dague_arena_elem_prefix_list_item_t {
    dague_list_item_t item;
    dague_arena_elem_prefix_t prefix;
};
/* for SSE, 16 is mandatory, most cache are 64 bit aligned */
#define DAGUE_ARENA_ALIGNMENT_64b 8
#define DAGUE_ARENA_ALIGNMENT_SSE 16
#define DAGUE_ARENA_ALIGNMENT_CL1 64
#define DAGUE_ARENA_MIN_ALIGNMENT(align) ((ptrdiff_t)(align*((sizeof(union _internal_dague_arena_elem_prefix_list_item_t)-align+1)/align+1)))

#define DAGUE_ARENA_IS_PTR(ptr) (((ptrdiff_t) ptr) & (ptrdiff_t) 1)
#define DAGUE_ARENA_PREFIX(ptr) ((dague_arena_elem_prefix_t*)(((ptrdiff_t) ptr) & ~(ptrdiff_t) 1))
#define DAGUE_ARENA_PTR(ptr) ((void*) (DAGUE_ARENA_PREFIX(ptr)->data))
#define DAGUE_ARENA_DATA(ptr) (DAGUE_ARENA_IS_PTR(ptr) ? DAGUE_ARENA_PTR(ptr) : ptr)
#define ADATA(ptr) DAGUE_ARENA_DATA(ptr)

void dague_arena_construct(dague_arena_t* arena, size_t elem_size, size_t alignment);
void dague_arena_construct_full(dague_arena_t* arena, size_t elem_size, size_t alignment, int32_t max_used, int32_t max_released); 
void dague_arena_destruct(dague_arena_t* arena);

void* dague_arena_get(dague_arena_t* arena);
void  dague_arena_release(void* ptr);


static inline uint32_t dague_arena_ref(void* ptr)
{
    assert(DAGUE_ARENA_IS_PTR(ptr));
    return dague_atomic_inc_32b(&DAGUE_ARENA_PREFIX(ptr)->refcount);
}
#define DAGUE_ARENA_REF_DATA(ptr) (DAGUE_ARENA_IS_PTR(ptr) ? dague_arena_ref(ptr) : 1)
#define AREF(ptr) DAGUE_ARENA_REF_DATA(ptr)

static inline uint32_t dague_arena_unref(void* ptr)
{
    uint32_t ret;
    assert(DAGUE_ARENA_IS_PTR(ptr));
    ret = dague_atomic_dec_32b(&DAGUE_ARENA_PREFIX(ptr)->refcount);
    if(0 == ret)
    {
        dague_arena_release(ptr);
    }
    return ret;
}
#define DAGUE_ARENA_UNREF_DATA(ptr) (DAGUE_ARENA_IS_PTR(ptr) ? dague_arena_unref(ptr) : 1)
#define AUNREF(ptr) DAGUE_ARENA_UNREF_DATA(ptr)


#endif /* __USE_ARENA_H__ */

