/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/arena.h"
#include "parsec/class/lifo.h"
#include "parsec/data_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/papi_sde.h"
#include <limits.h>

#if defined(PARSEC_PROF_TRACE_ACTIVE_ARENA_SET)

#include "profiling.h"

/* The user wants the amount of memory logged in the trace for each
 * arena malloc/free, we also expose it as a counter through PAPI-SDE
 * if another tool uses that info and PARSEC_PAPI_SDE is enabled */

extern int arena_memory_alloc_key, arena_memory_free_key;
extern int arena_memory_used_key, arena_memory_unused_key;
#define TRACE_MALLOC(key, size, ptr) do {                               \
        parsec_profiling_ts_trace_flags(key, (uint64_t)ptr, PROFILE_OBJECT_ID_NULL, \
                                        &size, PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO); \
        PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_MEM_ALLOC, size);   \
    } while(0)
#define TRACE_FREE(key, size, ptr) do {                                 \
        parsec_profiling_ts_trace_flags(key, (uint64_t)ptr, PROFILE_OBJECT_ID_NULL, \
                                        NULL, PARSEC_PROFILING_EVENT_COUNTER); \
        PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_MEM_ALLOC, -size);  \
    } while(0)

#else

/* The user does not want to log the amount of memory used for each
 * malloc/free; we just expose the current value to external tools
 * through PAPI-SDE. NB: if PARSEC_PAPI_SDE is not defined, this
 * translates into no-op */

#define TRACE_MALLOC(key, size, ptr) do {                               \
        PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_MEM_ALLOC, size);   \
    } while(0)
#define TRACE_FREE(key, size, ptr) do {                                 \
        PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_MEM_ALLOC, -size);  \
    } while(0)

#endif

#define PARSEC_ARENA_MIN_ALIGNMENT(align) ((ptrdiff_t)(align*((sizeof(parsec_arena_chunk_t)-1)/align+1)))

size_t parsec_arena_max_allocated_memory = SIZE_MAX;  /* unlimited */
size_t parsec_arena_max_cached_memory    = 256*1024*1024; /* limited to 256MB */


int parsec_arena_construct_ex(parsec_arena_t* arena,
                             size_t elem_size,
                             size_t alignment,
                             size_t max_allocated_memory,
                             size_t max_cached_memory)
{
    arena->elem_size = 0;  /* make sure the arena is marked as uninitialized to allow
                              the destructor to skip the lifo destruction. */
    /* alignment must be more than zero and power of two */
    if( (alignment <= 1) || (alignment & (alignment - 1)) )
        return PARSEC_ERR_BAD_PARAM;

    /* avoid dividing by zero */
    if( elem_size == 0 )
        return PARSEC_ERR_BAD_PARAM;

    assert(0 == (((uintptr_t)arena) % sizeof(uintptr_t))); /* is it aligned */

    PARSEC_OBJ_CONSTRUCT(&arena->area_lifo, parsec_lifo_t);
    arena->alignment    = alignment;
    arena->elem_size    = elem_size;
    arena->used         = 0;
    arena->max_used     = (max_allocated_memory / elem_size > (size_t)INT32_MAX)? INT32_MAX: max_allocated_memory / elem_size;
    arena->released     = 0;
    arena->max_released = (max_cached_memory / elem_size > (size_t)INT32_MAX)? INT32_MAX: max_cached_memory / elem_size;
    arena->data_malloc  = parsec_data_allocate;
    arena->data_free    = parsec_data_free;
    return PARSEC_SUCCESS;
}

int parsec_arena_construct(parsec_arena_t* arena,
                          size_t elem_size,
                          size_t alignment)
{
    return parsec_arena_construct_ex(arena, elem_size,
                                    alignment,
                                    parsec_arena_max_allocated_memory,
                                    parsec_arena_max_cached_memory);
}

static void parsec_arena_destructor(parsec_arena_t* arena)
{
    parsec_list_item_t* item;

    assert( arena->used == arena->released
         || arena->max_released == 0
         || arena->max_released == INT32_MAX
         || arena->max_used == 0
         || arena->max_used == INT32_MAX );

    /* If elem_size == 0, the arena has not been initialized */
    if ( 0 != arena->elem_size ) {
        while(NULL != (item = parsec_lifo_pop(&arena->area_lifo))) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Arena:\tfree element base ptr %p, data ptr %p (from arena %p)",
                                item, ((parsec_arena_chunk_t*)item)->data, arena);
            TRACE_FREE(arena_memory_free_key, -arena->elem_size, item);
            arena->data_free(item);
        }
        PARSEC_OBJ_DESTRUCT(&arena->area_lifo);
    }
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_arena_t, parsec_object_t, NULL, parsec_arena_destructor);

static inline parsec_list_item_t*
parsec_arena_get_chunk( parsec_arena_t *arena, size_t size, parsec_data_allocate_t alloc )
{
    parsec_lifo_t *list = &arena->area_lifo;
    parsec_list_item_t *item;
    item = parsec_lifo_pop(list);
    if( NULL != item ) {
        if( arena->max_released != INT32_MAX )
            (void)parsec_atomic_fetch_dec_int32(&arena->released);
    }
    else {
        if(arena->max_used != INT32_MAX) {
            int32_t current = parsec_atomic_fetch_inc_int32(&arena->used) + 1;
            if(current > arena->max_used) {
                (void)parsec_atomic_fetch_dec_int32(&arena->used);
                return NULL;
            }
        }
        if( size < sizeof( parsec_list_item_t ) )
            size = sizeof( parsec_list_item_t );
        item = (parsec_list_item_t *)alloc( size );
        TRACE_MALLOC(arena_memory_alloc_key, size, item);
        PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
        assert(NULL != item);
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tpop a data of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)",
                arena->elem_size, arena, arena->alignment, item, ((parsec_arena_chunk_t*)item)->data, sizeof(parsec_arena_chunk_t),
                PARSEC_ARENA_MIN_ALIGNMENT(arena->alignment));
    return item;
}

static void
parsec_arena_release_chunk(parsec_arena_t* arena,
                          parsec_arena_chunk_t *chunk)
{
    TRACE_FREE(arena_memory_unused_key, -arena->elem_size*chunk->count, chunk);

    if( (chunk->count == 1) && (arena->released < arena->max_released) ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tpush a data of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)",
                arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(parsec_arena_chunk_t),
                PARSEC_ARENA_MIN_ALIGNMENT(arena->alignment));
        if(arena->max_released != INT32_MAX) {
            (void)parsec_atomic_fetch_inc_int32(&arena->released);
        }
        parsec_lifo_push(&arena->area_lifo, &chunk->item);
        return;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tdeallocate a tile of size %zu x %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)",
            arena->elem_size, chunk->count, arena, arena->alignment, chunk, chunk->data, sizeof(parsec_arena_chunk_t),
            PARSEC_ARENA_MIN_ALIGNMENT(arena->alignment));
    TRACE_FREE(arena_memory_free_key, -arena->elem_size*chunk->count, chunk);
    if(arena->max_used != 0 && arena->max_used != INT32_MAX)
        (void)parsec_atomic_fetch_sub_int32(&arena->used, chunk->count);
    arena->data_free(chunk);
}

int  parsec_arena_allocate_device_private(parsec_data_copy_t *copy,
                                          parsec_arena_t *arena,
                                          size_t count, int device,
                                          parsec_datatype_t dtt)
{
    parsec_arena_chunk_t *chunk;
    parsec_data_t *data = copy->original;
    size_t size;

    assert(device == copy->device_index);
    (void)device;

    if( count == 1 ) {
        size = PARSEC_ALIGN(arena->elem_size + arena->alignment + sizeof(parsec_arena_chunk_t),
                            arena->alignment, size_t);
        chunk = (parsec_arena_chunk_t *)parsec_arena_get_chunk( arena, size, arena->data_malloc );
    } else {
        assert(count > 1);
        if(arena->max_used != INT32_MAX) {
            int32_t current = parsec_atomic_fetch_add_int32(&arena->used, count) + count;
            if(current > arena->max_used) {
                (void)parsec_atomic_fetch_sub_int32(&arena->used, count);
                return PARSEC_ERR_OUT_OF_RESOURCE;
            }
        }
        size = PARSEC_ALIGN(arena->elem_size * count + arena->alignment + sizeof(parsec_arena_chunk_t),
                            arena->alignment, size_t);
        chunk = (parsec_arena_chunk_t*)arena->data_malloc(size);
        PARSEC_OBJ_CONSTRUCT(&chunk->item, parsec_list_item_t);

        TRACE_MALLOC(arena_memory_alloc_key, size, chunk);
    }
    if(NULL == chunk) return PARSEC_ERR_OUT_OF_RESOURCE;  /* no more */

#if defined(PARSEC_DEBUG_PARANOID)
    PARSEC_LIST_ITEM_SINGLETON( &chunk->item );
#endif
    TRACE_MALLOC(arena_memory_used_key, size, chunk);

    chunk->origin = arena;
    chunk->count = count;
    chunk->data = PARSEC_ALIGN_PTR( ((ptrdiff_t)chunk + sizeof(parsec_arena_chunk_t)),
                                    arena->alignment, void* );

    assert(0 == (((ptrdiff_t)chunk->data) % arena->alignment));
    assert((arena->elem_size + (ptrdiff_t)chunk->data)  <= (size + (ptrdiff_t)chunk));

    data->nb_elts = count * arena->elem_size;

    copy->flags = PARSEC_DATA_FLAG_ARENA |
                  PARSEC_DATA_FLAG_PARSEC_OWNED |
                  PARSEC_DATA_FLAG_PARSEC_MANAGED;
    copy->dtt = dtt;
    copy->device_private = chunk->data;
    copy->arena_chunk = chunk;

    return PARSEC_SUCCESS;
}

parsec_data_copy_t *parsec_arena_get_copy(parsec_arena_t *arena,
                                          size_t count, int device,
                                          parsec_datatype_t dtt)
{
    parsec_data_t *data;
    parsec_data_copy_t *copy;
    int rc;

    
    data = parsec_data_new();
    if( NULL == data ) {
        return NULL;
    }

    copy = parsec_data_copy_new( data, device, dtt,
                                 PARSEC_DATA_FLAG_ARENA |
                                 PARSEC_DATA_FLAG_PARSEC_OWNED |
                                 PARSEC_DATA_FLAG_PARSEC_MANAGED);

    if(NULL == copy) {
        PARSEC_OBJ_RELEASE(data);
        return NULL;
    }

    rc = parsec_arena_allocate_device_private(copy, arena, count, device, dtt);

    /* This data is going to be released once all copies are released
     * It does not exist without at least a copy, and we don't give the
     * pointer to the user, so we must remove our retain from it
     */
    PARSEC_OBJ_RELEASE(data);

    if( PARSEC_SUCCESS != rc ) {
        PARSEC_OBJ_RELEASE(copy);
        return NULL;
    }

    return copy;
}

void parsec_arena_release(parsec_data_copy_t* copy)
{
    parsec_data_t *data;
    parsec_arena_chunk_t *chunk;
    parsec_arena_t* arena;

    data  = copy->original;
    chunk = copy->arena_chunk;
    arena = chunk->origin;

    assert(NULL != arena);
    assert(0 == (((uintptr_t)arena)%sizeof(uintptr_t))); /* is it aligned */

    if( NULL != data )
        parsec_data_copy_detach( data, copy, 0 );

    parsec_arena_release_chunk(arena, chunk);
}
