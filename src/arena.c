/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "arena.h"
#include "atomic.h"
#include "lifo.h"

#if 0
/* types used to compute alignment  */
union _internal_dague_arena_elem_prefix_list_item_t {
    dague_list_item_t item;
    dague_arena_elem_prefix_t prefix;
}
#define DAGUE_ARENA_MIN_ALIGNMENT(align) (align*((sizeof(union _internal_dague_arena_elem_prefix_list_item_t)-align+1)/align+1))
#endif 

void dague_arena_construct(dague_arena_t* arena, size_t elem_size, size_t alignment)
{
   dague_atomic_lifo_construct(&arena->lifo);
   arena->alignment = alignment;
   arena->elem_size = elem_size;
   arena->used = 0;     
   arena->released = 0;
   arena->max_used = INT32_MAX;
   arena->max_released = INT32_MAX;
   arena->malloc = NULL;
   arena->free = NULL;
}

void dague_arena_construct_full(dague_arena_t* arena, size_t elem_size, size_t alignment, int32_t max_used, int32_t max_released)
{
    dague_arena_construct(arena, elem_size, alignment);
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

void* dague_arena_get(dague_arena_t* arena)
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
        ptrdiff_t iptr = (ptrdiff_t) item;
        assert(NULL != item);
        assert(!(iptr & (ptrdiff_t)1)); /* all pointers are even */
        ptrdiff_t ialign = arena->alignment - 1;
        ptrdiff_t aptr = ((iptr+DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment)) & ~ialign);
            if(DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment) > (aptr - iptr))
            {
                aptr += DAGUE_ARENA_MIN_ALIGNMENT(arena->alignment);
                if(aptr + arena->elem_size > iptr + size)
                {
                    /*spilling, redo it with a bigger malloc*/
                    if(arena->free) arena->free(item);
                    else free(item);
                    size += ialign;
                    goto redo;
                }
            }
        dague_arena_elem_prefix_t* chunk = (void*) item;
        chunk->origin = arena;
        chunk->data = (void*) aptr;
        chunk->refcount = 1;
    }
    return (void*) (((ptrdiff_t) item) & 1);
}

void  dague_arena_release(void* ptr)
{
    dague_arena_elem_prefix_t* chunk = DAGUE_ARENA_PREFIX(ptr);
    assert(DAGUE_ARENA_IS_PTR(ptr));
    dague_arena_t* arena = chunk->origin;
    assert(NULL != chunk->origin);
    assert(1 == chunk->refcount);

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
