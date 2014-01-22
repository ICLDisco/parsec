/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <dague_config.h>
#include "arena.h"
#include "lifo.h"
#include "data.h"

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

    OBJ_CONSTRUCT(&arena->area_lifo, dague_lifo_t);
    arena->alignment = alignment;
    arena->elem_size = elem_size;
    arena->opaque_dtt = opaque_dtt;
    arena->data_malloc = dague_data_allocate;
    arena->data_free = dague_data_free;
    return 0;
}

void dague_arena_destruct(dague_arena_t* arena)
{
    dague_list_item_t* item;

    assert(0 == arena->used);

    while(NULL != (item = dague_lifo_pop(&arena->area_lifo))) {
        DEBUG3(("Arena:\tfree element base ptr %p, data ptr %p (from arena %p)\n",
                item, ((dague_arena_chunk_t*)item)->data, arena));
        arena->data_free(item);
    }
    OBJ_DESTRUCT(&arena->area_lifo);
}

static inline dague_list_item_t*
freelist_pop_or_create( dague_lifo_t *list, size_t size, dague_data_allocate_t alloc )
{
    dague_list_item_t *item;
    item = dague_lifo_pop(list);
    if( NULL == item ) {
        if( size < sizeof( dague_list_item_t ) )
            size = sizeof( dague_list_item_t );
        item = (dague_list_item_t *)alloc( size );
        OBJ_CONSTRUCT(item, dague_list_item_t);
        assert(NULL != item);
    }
    return item;
}

dague_data_t*
dague_arena_get(dague_arena_t* arena, size_t count)
{
    dague_arena_chunk_t* chunk;
    dague_data_t* data;
    dague_data_copy_t *copy;
    size_t size;

    data = dague_data_new();
    if( count == 1 ) {
        size = DAGUE_ALIGN(arena->elem_size + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);
        chunk = (dague_arena_chunk_t *)freelist_pop_or_create( &arena->area_lifo, size, arena->data_malloc );
    } else {
        assert(count > 1);
        size = DAGUE_ALIGN(arena->elem_size * count + arena->alignment + sizeof(dague_arena_chunk_t),
                           arena->alignment, size_t);
        chunk = (dague_arena_chunk_t*)arena->data_malloc(size);
    }
    data->nb_elts = count * arena->elem_size;

#if defined(DAGUE_DEBUG)
    DAGUE_LIST_ITEM_SINGLETON( &chunk->item );
#endif

    chunk->origin = arena;
    chunk->count = count;
    chunk->data = DAGUE_ALIGN_PTR( ((ptrdiff_t)chunk + sizeof(dague_arena_chunk_t)),
                                   arena->alignment, void* );

    assert(0 == (((ptrdiff_t)chunk->data) % arena->alignment));
    assert((arena->elem_size + (ptrdiff_t)chunk->data)  <= (size + (ptrdiff_t)chunk));

    copy = dague_data_copy_new( data, 0 );
    copy->flags |= DAGUE_DATA_FLAG_ARENA;
    copy->device_private = chunk->data;
    copy->arena_chunk = chunk;

    return data;
}

void dague_arena_release(dague_data_copy_t* copy)
{
    dague_data_t *data;
    dague_arena_chunk_t *chunk;
    dague_arena_t* arena;

    data = copy->original;
    chunk = copy->arena_chunk;
    arena = chunk->origin;

    assert(NULL != arena);
    assert(0 == (((uintptr_t)arena)%sizeof(uintptr_t))); /* is it aligned */

    dague_data_copy_detach( data, copy, 0 );
    if(chunk->count > 1 || arena->released >= arena->max_released) {
        DEBUG2(("Arena:\tdeallocate a tile of size %zu x %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                arena->elem_size, chunk->count, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t),
                DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        arena->data_free(chunk);
    } else {
        DEBUG2(("Arena:\tpush a data of size %zu from arena %p, aligned by %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n",
                arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(dague_arena_chunk_t),
                DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
        if(INT32_MAX != arena->max_released) {
            dague_atomic_inc_32b((uint32_t*)&arena->released);
        }
        dague_lifo_push(&arena->area_lifo, &chunk->item);
    }
  //  OBJ_RELEASE( data );
}
