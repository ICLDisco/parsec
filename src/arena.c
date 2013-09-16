/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "arena.h"
#include "atomic.h"
#include "lifo.h"

#define DAGUE_ARENA_MIN_ALIGNMENT(align) ((ptrdiff_t)(align*((sizeof(dague_arena_chunk_t)-1)/align+1)))

int dague_arena_construct(dague_arena_t* arena,
                          size_t elem_size,
                          size_t alignment,
                          dague_datatype_t opaque_dtt)
{
    /* alignment must be more than zero and power of two */
    if( (alignment <= 1) || (alignment & (alignment - 1)) )
        return -1;

    assert(0 == (((uintptr_t)arena) % sizeof(uintptr_t))); /* is it aligned */

    dague_lifo_construct(&arena->lifo);
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
                             dague_datatype_t opaque_dtt,
                             int32_t max_used,
                             int32_t max_released)
{
    /* alignment must be more than zero and power of two */
    if( (alignment <= 1) || (alignment & (alignment - 1)) )
        return -1;

    assert(0 == (((uintptr_t)arena) % sizeof(uintptr_t))); /* is it aligned */

    dague_arena_construct(arena, elem_size, alignment, opaque_dtt);
    arena->max_used = max_used;
    arena->max_released = max_released;
    return 0;
}

void dague_arena_destruct(dague_arena_t* arena)
{
    dague_list_item_t* item;

    assert(0 == arena->used);

    while(NULL != (item = dague_lifo_pop(&arena->lifo))) {
        arena->data_free(item);
    }
    dague_lifo_destruct(&arena->lifo);
}

dague_arena_chunk_t* dague_arena_get(dague_arena_t* arena, size_t count)
{
    dague_list_item_t* item;
    dague_arena_chunk_t* chunk;
    size_t size;

    if( count == 1 ) {
        size = DAGUE_ALIGN(arena->elem_size + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);

        if(arena->max_used != INT32_MAX) {
            dague_atomic_inc_32b((uint32_t*)&arena->used);
            if(arena->used > arena->max_used) {
                dague_atomic_dec_32b((uint32_t*)&arena->used);
                return NULL;
            }
        }

        item = dague_lifo_pop(&arena->lifo);
        if(NULL != item) {
            if(INT32_MAX != arena->max_released) {
                dague_atomic_dec_32b((uint32_t*)&arena->released);
                assert(arena->released >= 0);
            }
            DEBUG3(("Arena:\tretrieve a new tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                    arena->elem_size, arena, arena->alignment, item,
                    DAGUE_ALIGN_PTR( ((ptrdiff_t)item + sizeof(dague_arena_chunk_t)),
                                     arena->alignment, void* ),
                    sizeof(dague_arena_chunk_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        } else {
            item = arena->data_malloc(size);
            dague_list_item_construct(item);
            assert(NULL != item);
            DEBUG3(("Arena:\tallocate a new tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                    arena->elem_size, arena, arena->alignment, item,
                    DAGUE_ALIGN_PTR( ((ptrdiff_t)item + sizeof(dague_arena_chunk_t)),
                                     arena->alignment, void* ),
                    sizeof(dague_arena_chunk_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        }
        chunk = (dague_arena_chunk_t*) item;
    } else {
        assert(count > 1);
        size = DAGUE_ALIGN(arena->elem_size * count + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);
        chunk = (dague_arena_chunk_t*)arena->data_malloc( size );
    }

#if defined(DAGUE_DEBUG)
    DAGUE_LIST_ITEM_SINGLETON( (dague_list_item_t*)chunk );
#endif

    chunk->origin = arena;
    chunk->refcount = 1;
    chunk->count = count;
    chunk->data = DAGUE_ALIGN_PTR( ((ptrdiff_t)chunk + sizeof(dague_arena_chunk_t)),
                                   arena->alignment, void* );
    assert(0 == (((ptrdiff_t)chunk->data) % arena->alignment));
    assert((arena->elem_size + (ptrdiff_t)chunk->data)  <= (size + (ptrdiff_t)chunk));

    return (dague_arena_chunk_t*) (((ptrdiff_t) chunk) | (ptrdiff_t)1);
}

dague_arena_chunk_t* dague_arena_nolock_get(dague_arena_t* arena, size_t count)
{
    dague_list_item_t* item;
    dague_arena_chunk_t* chunk;
    size_t size;

    if( count == 1 ) {
        size = DAGUE_ALIGN(arena->elem_size + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);

        if(arena->max_used != INT32_MAX) {
            arena->used++;
            if(arena->used > arena->max_used) {
                arena->used--;
                return NULL;
            }
        }

        item = dague_lifo_nolock_pop(&arena->lifo);
        if(NULL != item) {
            if(INT32_MAX != arena->max_released) {
                arena->released--;
                assert(arena->released >= 0);
            }
            DEBUG3(("Arena:\tretrieve a new tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                    arena->elem_size, arena, arena->alignment, item,
                    DAGUE_ALIGN_PTR( ((ptrdiff_t)item + sizeof(dague_arena_chunk_t)),
                                     arena->alignment, void* ),
                    sizeof(dague_arena_chunk_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        } else {
            item = arena->data_malloc(size);
            dague_list_item_construct(item);
            assert(NULL != item);
            DEBUG3(("Arena:\tallocate a new tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                    arena->elem_size, arena, arena->alignment, item,
                    DAGUE_ALIGN_PTR( ((ptrdiff_t)item + sizeof(dague_arena_chunk_t)),
                                     arena->alignment, void* ),
                    sizeof(dague_arena_chunk_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        }
        chunk = (dague_arena_chunk_t*) item;
    } else {
        assert(count > 1);
        size = DAGUE_ALIGN(arena->elem_size * count + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);
        chunk = (dague_arena_chunk_t*)arena->data_malloc( size );
        item = (dague_list_item_t*)chunk; /* for the assert */
    }

#if defined(DAGUE_DEBUG)
    DAGUE_LIST_ITEM_SINGLETON( (dague_list_item_t*)chunk );
#endif

    chunk->origin = arena;
    chunk->refcount = 1;
    chunk->count = count;
    chunk->data = DAGUE_ALIGN_PTR( ((ptrdiff_t)chunk + sizeof(dague_arena_chunk_t)),
                                   arena->alignment, void* );
    assert(0 == (((ptrdiff_t)chunk->data) % arena->alignment));
    assert((arena->elem_size + (ptrdiff_t)chunk->data)  <= (size + (ptrdiff_t)item));

    return (dague_arena_chunk_t*) (((ptrdiff_t) chunk) | 1);
}



void dague_arena_release(dague_arena_chunk_t* ptr)
{
    dague_arena_chunk_t* chunk = DAGUE_ARENA_PREFIX(ptr);
    assert(DAGUE_ARENA_IS_PTR(ptr));
    dague_arena_t* arena = chunk->origin;
    assert(NULL != arena);
    assert(0 == (((uintptr_t)arena)%sizeof(uintptr_t))); /* is it aligned */
    assert(0 == chunk->refcount);

    if(chunk->count > 1 || arena->released >= arena->max_released) {
        DEBUG3(("Arena:\tdeallocate a tile of size %zu x %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                arena->elem_size, chunk->count, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t),
                DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        arena->data_free(chunk);
    } else {
        DEBUG3(("Arena:\tpush a tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
               arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        if(INT32_MAX != arena->max_released) {
            dague_atomic_inc_32b((uint32_t*)&arena->released);
        }
        dague_lifo_push(&arena->lifo, &chunk->item);
    }
    if(INT32_MAX != arena->max_used) {
        dague_atomic_dec_32b((uint32_t*)&arena->used);
        assert(0 <= arena->used);
    }
}


void dague_arena_nolock_release(dague_arena_chunk_t* ptr)
{
    dague_arena_chunk_t* chunk = DAGUE_ARENA_PREFIX(ptr);
    assert(DAGUE_ARENA_IS_PTR(ptr));
    dague_arena_t* arena = chunk->origin;
    assert(NULL != arena);
    assert(0 == (((uintptr_t)arena)%sizeof(uintptr_t))); /* is it aligned */
    assert(0 == chunk->refcount);

    if(chunk->count > 1 || arena->released >= arena->max_released) {
        DEBUG3(("Arena:\tdeallocate a tile of size %zu x %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                arena->elem_size, chunk->count, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t),
                DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        arena->data_free(chunk);
    } else {
        DEBUG3(("Arena:\tpush a tile of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
               arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        if(INT32_MAX != arena->max_released) {
            arena->released++;
        }
        dague_lifo_nolock_push(&arena->lifo, &chunk->item);
    }
    if(INT32_MAX != arena->max_used) {
        arena->used--;
        assert(0 <= arena->used);
    }
}
