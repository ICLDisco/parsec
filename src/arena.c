/*
 * Copyright (c) 2009      The University of Tennessee and The University
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

void dague_arena_construct(dague_arena_t* arena, size_t elem_size, size_t alignment, dague_remote_dep_datatype_t* opaque_dtt)
{
   dague_atomic_lifo_construct(&arena->lifo);
   arena->alignment = alignment;
   arena->elem_size = elem_size;
   arena->opaque_dtt = opaque_dtt;
   arena->used = 0;     
   arena->released = 0;
   arena->max_used = INT32_MAX;
   arena->max_released = INT32_MAX;
   arena->malloc = NULL;
   arena->free = NULL;
}

void dague_arena_construct_full(dague_arena_t* arena, size_t elem_size, size_t alignment, dague_remote_dep_datatype_t* opaque_dtt, int32_t max_used, int32_t max_released)
{
    dague_arena_construct(arena, elem_size, alignment, opaque_dtt);
    arena->max_used = max_used;
    arena->max_released = max_released;
}

void dague_arena_destruct(dague_arena_t* arena)
{
    dague_list_item_t* item;
    
    assert(0 == arena->used);
    
    while(NULL != (item = dague_atomic_lifo_pop(&arena->lifo))) 
    {
        if(arena->free) arena->free(item);
        else free(item);
    }
}

dague_arena_chunk_t* dague_arena_get(dague_arena_t* arena)
{
    dague_list_item_t* item;

    if(arena->max_used != INT32_MAX)
    {
        dague_atomic_inc_32b((uint32_t*)&arena->used);
        if(arena->used > arena->max_used)
        {
            dague_atomic_dec_32b((uint32_t*)&arena->used);
            return NULL;
        }
    }

    item = dague_atomic_lifo_pop(&arena->lifo);
    if(NULL != item)
    {
        if((INT32_MAX != arena->max_released) || (0 != arena->max_released))
        {
            dague_atomic_dec_32b((uint32_t*)&arena->released);
            assert(arena->released >= 0);
        }
    }
    else
    {
        size_t size = arena->elem_size + DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment);
redo:
        if(arena->malloc) item = arena->malloc(size);
        else item = malloc(size);
        assert(NULL != item);
        ptrdiff_t optr = (ptrdiff_t) item;
        assert(!(optr & (ptrdiff_t)1)); /* all pointers are even */
        
		ptrdiff_t aptr = ((optr+
							sizeof(union _internal_chunk_prefix_t)+
							arena->alignment-1)/arena->alignment)*arena->alignment;
        if(aptr + arena->elem_size > optr + size)
        {
        	/*spilling, redo it with a bigger malloc*/
            if(arena->free) arena->free(item);
            else free(item);
            size += arena->alignment-1;
            goto redo;
        }
    }
    dague_arena_chunk_t* chunk = (dague_arena_chunk_t*) item;
    chunk->origin = arena;
	chunk->refcount = 1;
    chunk->data = (void*) (((((ptrdiff_t)item)+
							sizeof(union _internal_chunk_prefix_t)+
							arena->alignment-1)/arena->alignment)*arena->alignment);
    DEBUG(("Arena get a new tile of size %zu from arena %p, aligned onby %zu, base ptr %p, data ptr %p, sizeof prefix %zu(%zd)\n", arena->elem_size, arena, arena->alignment, chunk, chunk->data, sizeof(union _internal_chunk_prefix_t), DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)));
    return (dague_arena_chunk_t*) (((ptrdiff_t) chunk) | 1);
}

void dague_arena_release(dague_arena_chunk_t* ptr)
{
    dague_arena_chunk_t* chunk = DAGUE_ARENA_PREFIX(ptr);
    assert(DAGUE_ARENA_IS_PTR(ptr));
    dague_arena_t* arena = chunk->origin;
    assert(NULL != chunk->origin);
    assert(1 >= chunk->refcount);

    if(arena->released >= arena->max_released)
    {
        if(arena->free) arena->free(chunk);
        else free(chunk);
    }
    else
    {
        if(INT32_MAX != arena->max_released)
        {
            dague_atomic_inc_32b((uint32_t*)&arena->released);
        }
        dague_atomic_lifo_push(&arena->lifo, (dague_list_item_t*) chunk);
    }
    if(INT32_MAX != arena->max_used)
    {
        dague_atomic_dec_32b((uint32_t*)&arena->used);
        assert(0 <= arena->used);
    }
}
