/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __USE_ARENA_H__
#define __USE_ARENA_H__

#include <stdlib.h>

#include "dague_config.h"
#include "dague_internal.h"
#if defined(HAVE_STDDEF_H)
#include <stddef.h>
#endif  /* HAVE_STDDEF_H */
#include "debug.h"
#include "stats.h"

#include "atomic.h"
#include "lifo.h"

#include "remote_dep.h"

#define DAGUE_ALIGN(x,a,t) (((x)+((t)(a)-1)) & ~(((t)(a)-1)))
#define DAGUE_ALIGN_PTR(x,a,t) ((t)DAGUE_ALIGN((uintptr_t)x, a, uintptr_t))
#define DAGUE_ALIGN_PAD_AMOUNT(x,s) ((~((uintptr_t)(x))+1) & ((uintptr_t)(s)-1))

struct dague_arena_t
{
    dague_lifo_t lifo;
    size_t alignment;                        /* alignment to be respected, elem_size should be >> alignment, prefix size is the minimum alignment */
    size_t elem_size;                        /* size of one element (unpacked in memory, aka extent) */
    dague_datatype_t opaque_dtt;             /* the appropriate type for the network engine to send an element */
    volatile int32_t used;                   /* elements currently out of the arena */
    int32_t max_used;                        /* maximum size of the arena in elements */
    volatile int32_t released;               /* elements currently not used but allocated */
    int32_t max_released;                    /* when more that max elements are released, they are really freed instead of joining the lifo
                                              * some host hardware requires special allocation functions (Cuda, pinning,
                                              * Open CL, ...). Defaults are to use C malloc/free */
    dague_data_allocate_t data_malloc;
    dague_data_free_t data_free;
};

struct dague_arena_chunk_t {
    dague_list_item_t item;                  /* chaining of this chunk when in an arena's free list.
                                              *   SINGLETON when ( (not in free list) and (in debug mode) ) */
    dague_arena_t* origin;
    void* data;
    uint32_t refcount;
    uint32_t count;
};

/* for SSE, 16 is mandatory, most cache are 64 bit aligned */
#define DAGUE_ARENA_ALIGNMENT_64b 8
#define DAGUE_ARENA_ALIGNMENT_INT sizeof(int)
#define DAGUE_ARENA_ALIGNMENT_PTR sizeof(void*)
#define DAGUE_ARENA_ALIGNMENT_SSE 16
#define DAGUE_ARENA_ALIGNMENT_CL1 64

#define DAGUE_ARENA_IS_PTR(ptr) (((ptrdiff_t) ptr) & (ptrdiff_t) 1)
#define DAGUE_ARENA_PREFIX(ptr) ((dague_arena_chunk_t*)(((ptrdiff_t) ptr) & ~(ptrdiff_t) 1))
#define DAGUE_ARENA_PTR(ptr) ((void*) (DAGUE_ARENA_PREFIX(ptr)->data))
#define DAGUE_ARENA_DATA(ptr) (DAGUE_ARENA_IS_PTR(ptr) ? DAGUE_ARENA_PTR(ptr) : ptr)
#define ADATA(ptr) DAGUE_ARENA_DATA(ptr)

#define DAGUE_ARENA_DATA_SIZE(ptr) (DAGUE_ARENA_PREFIX(ptr)->elem_size)
#define DAGUE_ARENA_DATA_TYPE(ptr) (DAGUE_ARENA_PREFIX(ptr)->origin->opaque_dtt)

int dague_arena_construct(dague_arena_t* arena,
                          size_t elem_size,
                          size_t alignment,
                          dague_datatype_t opaque_dtt);
int dague_arena_construct_ex(dague_arena_t* arena,
                             size_t elem_size,
                             size_t alignment,
                             dague_datatype_t opaque_dtt,
                             int32_t max_used,
                             int32_t max_released); 
void dague_arena_destruct(dague_arena_t* arena);

dague_arena_chunk_t* dague_arena_get(dague_arena_t* arena, size_t count);
dague_arena_chunk_t* dague_arena_nolock_get(dague_arena_t* arena, size_t count);
#define dague_uarena_get(arena, count) dague_arena_nolock_get(arena, count)
void dague_arena_release(dague_arena_chunk_t* ptr);
void dague_arena_nolock_release(dague_arena_chunk_t* ptr);
#define dague_uarena_release(ptr) dague_arena_nolock_release(ptr)

static inline uint32_t dague_arena_ref(dague_arena_chunk_t* ptr) {
    assert(DAGUE_ARENA_IS_PTR(ptr));
    return dague_atomic_inc_32b(&DAGUE_ARENA_PREFIX(ptr)->refcount);
}
static inline uint32_t dague_arena_nolock_ref(dague_arena_chunk_t* ptr) {
    assert(DAGUE_ARENA_IS_PTR(ptr));
    return ++DAGUE_ARENA_PREFIX(ptr)->refcount;
}
#define dague_uarena_ref(ptr) dague_arena_nolock_ref(ptr)
#define DAGUE_ARENA_REF_DATA(ptr) (DAGUE_ARENA_IS_PTR(ptr) ? dague_arena_ref(ptr) : 1)
#define AREF(ptr) DAGUE_ARENA_REF_DATA(ptr)

static inline uint32_t dague_arena_unref(dague_arena_chunk_t* ptr) {
    uint32_t ret;
    assert(DAGUE_ARENA_IS_PTR(ptr));
    ret = dague_atomic_dec_32b(&DAGUE_ARENA_PREFIX(ptr)->refcount);
    if(0 == ret) {
        dague_arena_release(ptr);
    }
    return ret;
}
static inline uint32_t dague_arena_nolock_unref(dague_arena_chunk_t* ptr) {
    uint32_t ret;
    assert(DAGUE_ARENA_IS_PTR(ptr));
    ret = --DAGUE_ARENA_PREFIX(ptr)->refcount;
    if(0 == ret) {
        dague_arena_nolock_release(ptr);
    }
    return ret;
}
#define dague_uarena_unref(ptr) dague_arena_nolock_unref(ptr)
#define DAGUE_ARENA_UNREF_DATA(ptr) (DAGUE_ARENA_IS_PTR(ptr) ? dague_arena_unref(ptr) : 1)
#define AUNREF(ptr) DAGUE_ARENA_UNREF_DATA(ptr)


#endif /* __USE_ARENA_H__ */

