/*
 * Copyright (c) 2010-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
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
        parsec_profiling_ts_trace_flags_info_fn(key, (uint64_t)ptr, PROFILE_OBJECT_ID_NULL, \
                                                memcpy, &size, PARSEC_PROFILING_EVENT_COUNTER|PARSEC_PROFILING_EVENT_HAS_INFO); \
        PARSEC_PAPI_SDE_COUNTER_ADD(PARSEC_PAPI_SDE_MEM_ALLOC, size);   \
    } while(0)
#define TRACE_FREE(key, size, ptr) do {                                 \
        parsec_profiling_ts_trace_flags_info_fn(key, (uint64_t)ptr, PROFILE_OBJECT_ID_NULL, \
                                                NULL, NULL, PARSEC_PROFILING_EVENT_COUNTER); \
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
    const char *allocation_error = "out of resource";
    item = parsec_lifo_pop(list);
    if( NULL != item ) {
        if( arena->max_released != INT32_MAX )
            (void)parsec_atomic_fetch_dec_int32(&arena->released);
    }
    else {
        if(arena->max_used != INT32_MAX) {
            int32_t current = parsec_atomic_fetch_inc_int32(&arena->used) + 1;
            if(current > arena->max_used) {
                allocation_error = "maximum allocation count reached";
                goto allocation_failed;
            }
        }
        if( size < sizeof( parsec_list_item_t ) )
            size = sizeof( parsec_list_item_t );
        item = (parsec_list_item_t *)alloc( size );
        if( NULL == item ) {
            allocation_error = "allocator returned NULL";
            goto allocation_failed;
        }
        TRACE_MALLOC(arena_memory_alloc_key, size, item);
        PARSEC_OBJ_CONSTRUCT(item, parsec_list_item_t);
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tpop a data of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)",
                arena->elem_size, arena, arena->alignment, item, ((parsec_arena_chunk_t*)item)->data, sizeof(parsec_arena_chunk_t),
                PARSEC_ARENA_MIN_ALIGNMENT(arena->alignment));
    return item;

  allocation_failed:
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                         "Arena:\tfailed to allocate data of size %zu from arena %p, aligned by %zu: %s (used %d max %d)",
                         arena->elem_size, arena, arena->alignment, allocation_error, arena->used, arena->max_used);
    if(arena->max_used != INT32_MAX) {
        (void)parsec_atomic_fetch_dec_int32(&arena->used);
    }
    (void)allocation_error;
    return NULL;
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

/*
 * Storage behind PARSEC_ARENA_AUTO. A request of up to parsec_arena_auto_max_bucket
 * bytes is served as a single element of the smallest power-of-two bucket that
 * holds it, so that the bucket arena can cache it for the next request of a
 * similar size. Anything larger is served, in grains of 64 bytes, by an arena
 * that never caches: a grain rather than a byte, so that the element counts the
 * arenas keep (int32_t) measure allocations well past 2GiB. Slot 0 is that
 * arena; slot i > 0 is the bucket of 2^i bytes. The arenas are created on
 * first use.
 */
size_t parsec_arena_auto_max_bucket = 1024*1024;  /* 1MiB */
size_t parsec_arena_auto_max_cached_memory = 256*1024*1024;  /* 256MB for all the buckets together */

#define PARSEC_ARENA_AUTO_MIN_SHIFT  6   /* 64 bytes */
#define PARSEC_ARENA_AUTO_MAX_SHIFT 30   /* 1GiB, hard limit on the bucket sizes */
#define PARSEC_ARENA_AUTO_GRAIN     64   /* bytes per element of the arena serving larger requests */

static parsec_arena_t *volatile parsec_arena_auto_arenas[PARSEC_ARENA_AUTO_MAX_SHIFT + 1];

/* The smallest bucket holding bytes, as its shift */
static inline int parsec_arena_auto_shift_of(size_t bytes)
{
    int shift;
    for( shift = PARSEC_ARENA_AUTO_MIN_SHIFT;
         (shift < PARSEC_ARENA_AUTO_MAX_SHIFT) && ((((size_t)1) << shift) < bytes);
         shift++ );
    return shift;
}

/*
 * The buckets share parsec_arena_auto_max_cached_memory: each of the buckets
 * parsec_arena_auto_max_bucket allows for is given an equal part of it, so
 * that together they never keep more than that cached whatever the sizes a
 * workload goes through.
 */
static size_t parsec_arena_auto_bucket_cache(void)
{
    int nb_buckets = parsec_arena_auto_shift_of(parsec_arena_auto_max_bucket) - PARSEC_ARENA_AUTO_MIN_SHIFT + 1;
    return parsec_arena_auto_max_cached_memory / (size_t)nb_buckets;
}

/* The arena of a slot, created on its first use: slot 0 is the arena of
 * grains, which caches nothing, slot i > 0 the bucket of 2^i bytes. */
static parsec_arena_t *
parsec_arena_auto_get(int slot)
{
    parsec_arena_t *arena = parsec_arena_auto_arenas[slot];
    size_t elem_size = (0 == slot) ? PARSEC_ARENA_AUTO_GRAIN : (((size_t)1) << slot);
    size_t max_cached = (0 == slot) ? 0 : parsec_arena_auto_bucket_cache();

    if( PARSEC_LIKELY(NULL != arena) ) {
        /* Pair with the CAS that published it: nothing of the arena is read
         * before the pointer that was installed after its construction. */
        parsec_atomic_rmb();
        return arena;
    }
    arena = PARSEC_OBJ_NEW(parsec_arena_t);
    if( NULL == arena ) {
        return NULL;
    }
    if( PARSEC_SUCCESS != parsec_arena_construct_ex(arena, elem_size, PARSEC_ARENA_ALIGNMENT_SSE,
                                                    parsec_arena_max_allocated_memory, max_cached) ) {
        PARSEC_OBJ_RELEASE(arena);
        return NULL;
    }
    if( !parsec_atomic_cas_ptr(&parsec_arena_auto_arenas[slot], NULL, arena) ) {
        PARSEC_OBJ_RELEASE(arena);  /* another thread installed this slot first */
        arena = parsec_arena_auto_arenas[slot];
        parsec_atomic_rmb();
    }
    return arena;
}

/*
 * Translate a PARSEC_ARENA_AUTO request for count elements of dtt into the
 * arena that serves it and the number of elements of that arena to allocate.
 * Returns the size of the request in bytes, or 0 if it cannot be served.
 */
static size_t
parsec_arena_auto_select(size_t count, parsec_datatype_t dtt,
                         parsec_arena_t **arena, size_t *arena_count)
{
    ptrdiff_t lb, extent;
    size_t bytes;
    int shift;

    if( (PARSEC_SUCCESS != parsec_type_extent(dtt, &lb, &extent)) || (extent <= 0) ) {
        return 0;
    }
    (void)lb;
    if( (0 == count) || (count > (SIZE_MAX / (size_t)extent)) ) {
        return 0;  /* nothing, or more than a size_t holds */
    }
    bytes = count * (size_t)extent;
    if( bytes <= parsec_arena_auto_max_bucket ) {
        shift = parsec_arena_auto_shift_of(bytes);
        if( (((size_t)1) << shift) >= bytes ) {
            *arena_count = 1;
            *arena = parsec_arena_auto_get(shift);
            return (NULL == *arena) ? 0 : bytes;
        }
    }
    *arena_count = bytes / PARSEC_ARENA_AUTO_GRAIN + ((0 != (bytes % PARSEC_ARENA_AUTO_GRAIN)) ? 1 : 0);
    *arena = parsec_arena_auto_get(0);
    return (NULL == *arena) ? 0 : bytes;
}

void parsec_arena_auto_fini(void)
{
    for( int slot = 0; slot <= PARSEC_ARENA_AUTO_MAX_SHIFT; slot++ ) {
        parsec_arena_t *arena = parsec_arena_auto_arenas[slot];
        if( NULL == arena ) continue;
        parsec_arena_auto_arenas[slot] = NULL;
        PARSEC_OBJ_RELEASE(arena);
    }
}

int  parsec_arena_allocate_device_private(parsec_data_copy_t *copy,
                                          parsec_arena_t *arena,
                                          size_t count, int device,
                                          parsec_datatype_t dtt)
{
    parsec_arena_chunk_t *chunk;
    parsec_data_t *data = copy->original;
    size_t size, auto_bytes = 0;
    const char *allocation_error = "out of resource"; (void)allocation_error;

    assert(device == copy->device_index);
    (void)device;

    if( PARSEC_ARENA_AUTO == arena ) {
        auto_bytes = parsec_arena_auto_select(count, dtt, &arena, &count);
        if( 0 == auto_bytes ) {
            return PARSEC_ERR_OUT_OF_RESOURCE;
        }
    }

    if( count == 1 ) {
        size = PARSEC_ALIGN(arena->elem_size + arena->alignment + sizeof(parsec_arena_chunk_t),
                            arena->alignment, size_t);
        chunk = (parsec_arena_chunk_t *)parsec_arena_get_chunk( arena, size, arena->data_malloc );
    } else {
        assert(count > 1);
        if(arena->max_used != INT32_MAX) {
            int32_t current = parsec_atomic_fetch_add_int32(&arena->used, count) + count;
            if(current > arena->max_used) {
                allocation_error = "maximum allocation count reached";
                goto allocation_failed;
            }
        }
        size = PARSEC_ALIGN(arena->elem_size * count + arena->alignment + sizeof(parsec_arena_chunk_t),
                            arena->alignment, size_t);
        chunk = (parsec_arena_chunk_t*)arena->data_malloc(size);
        if(NULL == chunk) {
            allocation_error = "allocator returned NULL";
            goto allocation_failed;
        }
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

    /* An AUTO request may land in a larger bucket: the data spans the request */
    data->span = (0 != auto_bytes) ? auto_bytes : count * arena->elem_size;

    copy->flags = PARSEC_DATA_FLAG_ARENA |
                  PARSEC_DATA_FLAG_PARSEC_OWNED |
                  PARSEC_DATA_FLAG_PARSEC_MANAGED;
    copy->dtt = dtt;
    copy->device_private = chunk->data;
    copy->arena_chunk = chunk;

    return PARSEC_SUCCESS;

  allocation_failed:
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                         "Arena:\tfailed to allocate %zu data items of size %zu from arena %p, aligned by %zu: %s (used %d max %d)",
                         count, arena->elem_size, arena, arena->alignment, allocation_error, arena->used, arena->max_used);
    if(arena->max_used != INT32_MAX) {
        (void)parsec_atomic_fetch_sub_int32(&arena->used, count);
    }
    return PARSEC_ERR_OUT_OF_RESOURCE;
}

#include "parsec/utils/zone_malloc.h"
#include "mca/device/device_gpu.h"

#if defined(PARSEC_DEBUG)
static int64_t parsec_countable_incoming_message = 0xF000000000000000;
#endif  /* defined(PARSEC_DEBUG) */

static inline parsec_data_copy_t *
parsec_arena_internal_copy_new(parsec_arena_t *arena,
                               parsec_data_t *data,
                               size_t count, int device,
                               parsec_datatype_t dtt)
{
    parsec_data_copy_t *copy = NULL;
    parsec_data_t* ldata = data;
    if( NULL == data ) {
        ldata = parsec_data_new();
        if( NULL == ldata ) {
            return NULL;
        }
#if defined(PARSEC_DEBUG)
        /* Name the data with a default key to facilitate debugging */
        ldata->key = (uint64_t)parsec_atomic_fetch_inc_int64(&parsec_countable_incoming_message);
        ldata->key |= ((uint64_t)device) << 56;
#endif  /* defined(PARSEC_DEBUG) */
    }
    if( 0 == device ) {
        copy = parsec_data_copy_new(ldata, device, dtt,
                                    PARSEC_DATA_FLAG_PARSEC_OWNED | PARSEC_DATA_FLAG_PARSEC_MANAGED | PARSEC_DATA_FLAG_ARENA);
        if (NULL == copy) {
            goto free_and_return;
        }
        int rc = parsec_arena_allocate_device_private(copy, arena, count, device, dtt);
        if (PARSEC_SUCCESS != rc) {
            goto free_and_return;
        }
        return copy;
    }
    /**
     * This part is not really nice, it breaks the separation between devices, and how their memory is
     * managed. But, it should give nice performance improvements if the communication layer is
     * capable of sending or receiving data directly to and from the accelerator memory. The only drawback
     * is that once the GPU memory is full, this will fail, so the software will fall back to the
     * prior behavior, going through the CPU memory.
     *
     * The deallocation is not symmetric: regular GPU task copies are released
     * by the GPU management code, while self-contained temporaries are released
     * with their owning parsec_data_t.
     */
    parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t *)parsec_mca_device_get(device);
    if (NULL == gpu_device) {
        goto free_and_return;
    }
    size_t size = count * arena->elem_size;
    void *device_private;
#if defined(PARSEC_GPU_ALLOC_PER_TILE)
    int rc = gpu_device->memory_allocate(gpu_device, size, &device_private);
    if( PARSEC_SUCCESS != rc ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tallocate data copy on device %d of size %zu failed (out of memory)\n",
                             device, size);
        goto free_and_return;
    }
#else
    device_private = zone_malloc(gpu_device->memory, size);
    if( NULL == device_private ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tallocate data copy on device %d of size %zu from zone %p failed (out of memory)\n",
                             device, size, (void *)gpu_device->memory);
        goto free_and_return;
    }
#endif
    copy = parsec_data_copy_new(ldata, device, dtt,
                                PARSEC_DATA_FLAG_PARSEC_OWNED | PARSEC_DATA_FLAG_PARSEC_MANAGED);
    if (NULL == copy) {
#if defined(PARSEC_GPU_ALLOC_PER_TILE)
        gpu_device->memory_free(gpu_device, device_private);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tallocate data copy on device %d of size %zu failed to allocate copy (out of memory)\n",
                             device, size);
#else
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tallocate data copy on device %d of size %zu from zone %p failed to allocate copy (out of memory)\n",
                             device, size, (void *)gpu_device->memory);
        zone_free(gpu_device->memory, device_private);
#endif
        goto free_and_return;
    }
    copy->dtt = dtt;
    copy->device_private = device_private;
#if defined(PARSEC_GPU_ALLOC_PER_TILE)
    copy->arena_chunk = NULL;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tallocate data copy on device %d of size %zu, data ptr %p",
                         device, size, (void*)copy->device_private);
#else
    copy->arena_chunk = (parsec_arena_chunk_t*)gpu_device->memory;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Arena:\tallocate data copy on device %d of size %zu from zone %p, data ptr %p",
                         device, size, (void*)copy->arena_chunk, (void*)copy->device_private);
#endif
    copy->version = 0;
    copy->coherency_state = PARSEC_DATA_COHERENCY_INVALID;
    copy->original->owner_device = device;
    copy->original->preferred_device = device;
    return copy;
  free_and_return:
    if( NULL != copy )
        PARSEC_OBJ_RELEASE(copy);
    if( NULL == data)
        PARSEC_OBJ_RELEASE(ldata); /* release the locally allocated data */
    return NULL;
}

parsec_data_copy_t *
parsec_arena_get_new_copy(parsec_arena_t *arena,
                          size_t count, int device,
                          parsec_datatype_t dtt)
{
    parsec_data_copy_t *dev0_copy, *copy;
    size_t auto_bytes = 0;

    if( PARSEC_ARENA_AUTO == arena ) {
        auto_bytes = parsec_arena_auto_select(count, dtt, &arena, &count);
        if( 0 == auto_bytes ) {
            return NULL;
        }
    }

    dev0_copy = parsec_arena_internal_copy_new(arena, NULL, count, 0 /* first allocate the copy on the device 0 */, dtt);
    if( NULL == dev0_copy ) {
        return NULL;
    }
    dev0_copy->coherency_state = PARSEC_DATA_COHERENCY_INVALID;
    dev0_copy->version = 0;  /* start from somewhere */
    if( 0 != auto_bytes ) {
        /* the bucket may be larger than the request, the data is not */
        dev0_copy->original->span = auto_bytes;
    }
    if( 0 == device ) {
        /* Device 0 (CPU) copy is special: it belongs to the user in the current design,
         * so we must not release the original here.
         */
        return dev0_copy;
    }

    copy = parsec_arena_internal_copy_new(arena, dev0_copy->original, count, device, dtt);
    if( NULL == copy ) {
        copy = dev0_copy;  /* return the main memory data copy */
    }
    /* This data is going to be released once all copies are released
     * It does not exist without at least a copy, and we don't give the
     * pointer to the user, so we must remove our retain from it
     */
    PARSEC_OBJ_RELEASE(dev0_copy->original);
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
