/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/arena.h"
#include "dague/class/lifo.h"
#include "dague/data_internal.h"
#include <limits.h>

#if defined(DAGUE_PROF_TRACE) && defined(DAGUE_PROF_TRACE_ACTIVE_ARENA_SET)

#include "profiling.h"

extern int arena_memory_alloc_key, arena_memory_free_key;
extern int arena_memory_used_key, arena_memory_unused_key;
#define TRACE_MALLOC(key, size, ptr) dague_profiling_ts_trace_flags(key, (uint64_t)ptr, PROFILE_OBJECT_ID_NULL,\
                                                                    &size, DAGUE_PROFILING_EVENT_COUNTER|DAGUE_PROFILING_EVENT_HAS_INFO)
#define TRACE_FREE(key, ptr)         dague_profiling_ts_trace_flags(key, (uint64_t)ptr, PROFILE_OBJECT_ID_NULL,\
                                                                    NULL, DAGUE_PROFILING_EVENT_COUNTER)
#else
#define TRACE_MALLOC(key, size, ptr) do {} while (0)
#define TRACE_FREE(key, ptr) do {} while (0)
#endif

#define DAGUE_ARENA_MIN_ALIGNMENT(align) ((ptrdiff_t)(align*((sizeof(dague_arena_chunk_t)-1)/align+1)))

size_t dague_arena_max_allocated_memory = SIZE_MAX;  /* unlimited */
size_t dague_arena_max_cached_memory    = 256*1024*1024; /* limited to 256MB */

int dague_arena_construct_ex(dague_arena_t* arena,
                             size_t elem_size,
                             size_t alignment,
                             dague_datatype_t opaque_dtt,
                             size_t max_allocated_memory,
                             size_t max_cached_memory)
{
    /* alignment must be more than zero and power of two */
    if( (alignment <= 1) || (alignment & (alignment - 1)) )
        return -1;

    /* avoid dividing by zero */
    if( elem_size == 0 )
        return -1;

    assert(0 == (((uintptr_t)arena) % sizeof(uintptr_t))); /* is it aligned */

    OBJ_CONSTRUCT(&arena->area_lifo, dague_lifo_t);
    arena->alignment    = alignment;
    arena->elem_size    = elem_size;
    arena->opaque_dtt   = opaque_dtt;
    arena->used         = 0;
    arena->max_used     = (max_allocated_memory / elem_size > (size_t)INT32_MAX)? INT32_MAX: max_allocated_memory / elem_size;
    arena->released     = 0;
    arena->max_released = (max_cached_memory / elem_size > (size_t)INT32_MAX)? INT32_MAX: max_cached_memory / elem_size;
    arena->data_malloc  = dague_data_allocate;
    arena->data_free    = dague_data_free;
    return 0;
}

int dague_arena_construct(dague_arena_t* arena,
                          size_t elem_size,
                          size_t alignment,
                          dague_datatype_t opaque_dtt)
{
    return dague_arena_construct_ex(arena, elem_size,
                                    alignment, opaque_dtt,
                                    dague_arena_max_allocated_memory,
                                    dague_arena_max_cached_memory);
}

void dague_arena_destruct(dague_arena_t* arena)
{
    dague_list_item_t* item;

    assert( arena->used == arena->released
         || arena->max_released == 0
         || arena->max_released == INT32_MAX
         || arena->max_used == 0
         || arena->max_used == INT32_MAX );

    while(NULL != (item = dague_lifo_pop(&arena->area_lifo))) {
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Arena:\tfree element base ptr %p, data ptr %p (from arena %p)",
                item, ((dague_arena_chunk_t*)item)->data, arena);
        TRACE_FREE(arena_memory_free_key, item);
        arena->data_free(item);
    }
    OBJ_DESTRUCT(&arena->area_lifo);
}

static inline dague_list_item_t*
dague_arena_get_chunk( dague_arena_t *arena, size_t size, dague_data_allocate_t alloc )
{
    dague_lifo_t *list = &arena->area_lifo;
    dague_list_item_t *item;
    item = dague_lifo_pop(list);
    if( NULL != item ) {
        if( arena->max_released != INT32_MAX ) dague_atomic_dec_32b((uint32_t*)&arena->released);
    }
    else {
        if(arena->max_used != INT32_MAX) {
            int32_t current = dague_atomic_add_32b(&arena->used, 1);
            if(current > arena->max_used) {
                dague_atomic_dec_32b((uint32_t*)&arena->used);
                return NULL;
            }
        }
        if( size < sizeof( dague_list_item_t ) )
            size = sizeof( dague_list_item_t );
        item = (dague_list_item_t *)alloc( size );
        TRACE_MALLOC(arena_memory_alloc_key, size, item);
        OBJ_CONSTRUCT(item, dague_list_item_t);
        assert(NULL != item);
    }
    return item;
}

static void
dague_arena_release_chunk(dague_arena_t* arena,
                          dague_arena_chunk_t *chunk)
{
    TRACE_FREE(arena_memory_unused_key, chunk);

    if( (chunk->count == 1) && (arena->released < arena->max_released) ) {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Arena:\tpush a data of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)",
                arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t),
                DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment));
        if(arena->max_released != INT32_MAX) {
            dague_atomic_inc_32b((uint32_t*)&arena->released);
        }
        dague_lifo_push(&arena->area_lifo, &chunk->item);
        return;
    }
    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Arena:\tdeallocate a tile of size %zu x %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)",
            arena->elem_size, chunk->count, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t),
            DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment));
    TRACE_FREE(arena_memory_free_key, chunk);
    if(arena->max_used != 0 && arena->max_used != INT32_MAX)
        dague_atomic_sub_32b(&arena->used, chunk->count);
    arena->data_free(chunk);
}

dague_data_copy_t *dague_arena_get_copy(dague_arena_t *arena, size_t count, int device)
{
    dague_arena_chunk_t *chunk;
    dague_data_t *data;
    dague_data_copy_t *copy;
    size_t size;

    if( count == 1 ) {
        size = DAGUE_ALIGN(arena->elem_size + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);
        chunk = (dague_arena_chunk_t *)dague_arena_get_chunk( arena, size, arena->data_malloc );
    } else {
        assert(count > 1);
        if(arena->max_used != INT32_MAX) {
            int32_t current = dague_atomic_add_32b(&arena->used, count);
            if(current > arena->max_used) {
                dague_atomic_sub_32b(&arena->used, count);
                return NULL;
            }
        }
        size = DAGUE_ALIGN(arena->elem_size * count + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);
        chunk = (dague_arena_chunk_t*)arena->data_malloc(size);
        OBJ_CONSTRUCT(&chunk->item, dague_list_item_t);
        chunk->refcount = 1;

        TRACE_MALLOC(arena_memory_alloc_key, size, chunk);
    }
    if(NULL == chunk) return NULL;  /* no more */

#if defined(DAGUE_DEBUG_PARANOID)
    DAGUE_LIST_ITEM_SINGLETON( &chunk->item );
#endif
    TRACE_MALLOC(arena_memory_used_key, size, chunk);

    chunk->origin = arena;
    chunk->count = count;
    chunk->data = DAGUE_ALIGN_PTR( ((ptrdiff_t)chunk + sizeof(dague_arena_chunk_t)),
                                   arena->alignment, void* );

    assert(0 == (((ptrdiff_t)chunk->data) % arena->alignment));
    assert((arena->elem_size + (ptrdiff_t)chunk->data)  <= (size + (ptrdiff_t)chunk));

    data = dague_data_new();
    if( NULL == data ) {
        dague_arena_release_chunk(arena, chunk);
        return NULL;
    }

    data->nb_elts = count * arena->elem_size;

    copy = dague_data_copy_new( data, device );
    copy->flags |= DAGUE_DATA_FLAG_ARENA;
    copy->device_private = chunk->data;
    copy->arena_chunk = chunk;

    /* This data is going to be released once all copies are released
     * It does not exist without at least a copy, and we don't give the
     * pointer to the user, so we must remove our retain from it
     */
    OBJ_RELEASE(data);

    return copy;
}

void dague_arena_release(dague_data_copy_t* copy)
{
    dague_data_t *data;
    dague_arena_chunk_t *chunk;
    dague_arena_t* arena;

    data  = copy->original;
    chunk = copy->arena_chunk;
    arena = chunk->origin;

    assert(NULL != arena);
    assert(0 == (((uintptr_t)arena)%sizeof(uintptr_t))); /* is it aligned */

    if( NULL != data )
        dague_data_copy_detach( data, copy, 0 );

    dague_arena_release_chunk(arena, chunk);
}
