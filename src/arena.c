/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "arena.h"
#include "atomic.h"
#include "lifo.h"

/* types used to compute alignment  */
union _internal_chunk_prefix_t {
    dague_list_item_t item;
    dague_arena_chunk_t prefix;
};
#define DAGUE_ARENA_MIN_ALIGNMENT(align) ((ptrdiff_t)(align*((sizeof(union _internal_chunk_prefix_t)-1)/align+1)))

int dague_arena_construct(dague_arena_t* arena,
                          size_t elem_size,
                          size_t alignment,
                          dague_remote_dep_datatype_t opaque_dtt)
{
    /* alignment must be more than zero and power of two */
    if( (alignment <= 1) || (alignment & (alignment - 1)) )
        return -1;

    dague_atomic_lifo_construct(&arena->lifo);
    arena->alignment = alignment;
    arena->elem_size = elem_size;
    arena->opaque_dtt = opaque_dtt;
    arena->used = 0;     
    arena->released = 0;
    arena->max_used = INT32_MAX;
    arena->max_released = INT32_MAX;
    arena->data_malloc = dague_data_allocate;
    arena->data_free = dague_data_free;
    return 0;
}

int dague_arena_construct_ex(dague_arena_t* arena,
                             size_t elem_size,
                             size_t alignment,
                             dague_remote_dep_datatype_t opaque_dtt,
                             int32_t max_used,
                             int32_t max_released)
{
    /* alignment must be more than zero and power of two */
    if( (alignment <= 1) || (alignment & (alignment - 1)) )
        return -1;

    dague_arena_construct(arena, elem_size, alignment, opaque_dtt);
    arena->max_used = max_used;
    arena->max_released = max_released;
    return 0;
}

void dague_arena_destruct(dague_arena_t* arena)
{
    dague_list_item_t* item;
    
    assert(0 == arena->used);
    
    while(NULL != (item = dague_atomic_lifo_pop(&arena->lifo))) {
        arena->data_free(item);
    }
}

dague_arena_chunk_t* dague_arena_get(dague_arena_t* arena)
{
    dague_list_item_t* item;
    dague_arena_chunk_t* chunk;
    size_t size = DAGUE_ALIGN(arena->elem_size + arena->alignment + sizeof(union _internal_chunk_prefix_t),
                              arena->alignment, size_t);

    if(arena->max_used != INT32_MAX) {
        dague_atomic_inc_32b((uint32_t*)&arena->used);
        if(arena->used > arena->max_used) {
            dague_atomic_dec_32b((uint32_t*)&arena->used);
            return NULL;
        }
    }

    item = dague_atomic_lifo_pop(&arena->lifo);
    if(NULL != item) {
        if(INT32_MAX != arena->max_released) {
            dague_atomic_dec_32b((uint32_t*)&arena->released);
            assert(arena->released >= 0);
        }
        DEBUG(("Arena retrieve a new tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
               arena->elem_size, arena, arena->alignment, item,
               DAGUE_ALIGN_PTR( ((ptrdiff_t)item + sizeof(union _internal_chunk_prefix_t)),
                                   arena->alignment, void* ),
               sizeof(union _internal_chunk_prefix_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
    } else {
        item = arena->data_malloc(size);
        assert(NULL != item);
        DEBUG(("Arena allocate a new tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
               arena->elem_size, arena, arena->alignment, item,
               DAGUE_ALIGN_PTR( ((ptrdiff_t)item + sizeof(union _internal_chunk_prefix_t)),
                                   arena->alignment, void* ),
               sizeof(union _internal_chunk_prefix_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
    }
    chunk = (dague_arena_chunk_t*) item;
    chunk->origin = arena;
    chunk->refcount = 0;
    chunk->data = DAGUE_ALIGN_PTR( ((ptrdiff_t)item + sizeof(union _internal_chunk_prefix_t)),
                                   arena->alignment, void* );
    assert(((unsigned char*)chunk->data + arena->elem_size) <= ((unsigned char*)item + size));
    return (dague_arena_chunk_t*) (((ptrdiff_t) chunk) | 1);
}

void dague_arena_release(dague_arena_chunk_t* ptr)
{
    dague_arena_chunk_t* chunk = DAGUE_ARENA_PREFIX(ptr);
    assert(DAGUE_ARENA_IS_PTR(ptr));
    dague_arena_t* arena = chunk->origin;
    assert(NULL != chunk->origin);
    assert(1 >= chunk->refcount);

    if(arena->released >= arena->max_released) {
        DEBUG(("Arena deallocate a tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
               arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(union _internal_chunk_prefix_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        arena->data_free(chunk);
    } else {
        DEBUG(("Arena push a tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
               arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(union _internal_chunk_prefix_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        if(INT32_MAX != arena->max_released) {
            dague_atomic_inc_32b((uint32_t*)&arena->released);
        }
        DAGUE_LIST_ITEM_SINGLETON(chunk);
        dague_atomic_lifo_push(&arena->lifo, (dague_list_item_t*) chunk);
    }
    if(INT32_MAX != arena->max_used) {
        dague_atomic_dec_32b((uint32_t*)&arena->used);
        assert(0 <= arena->used);
    }
}

