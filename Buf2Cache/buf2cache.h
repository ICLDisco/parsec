#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "atomic.h"

///////////////////////////////////////////////////////////////////////////////////////////////
// types

typedef struct{
    const void *tile_ptr;
    int64_t age;
    volatile int lock;
} cache_entry_t;


typedef struct _cache_t cache_t;

struct _cache_t{
    cache_t *parent;
    int tile_capacity;
    cache_entry_t *entries;
};
     

///////////////////////////////////////////////////////////////////////////////////////////////
// function forward declarations

cache_t *cache_create(int core_count, cache_t *parent, int tile_capacity);
const void *cache_buf_referenced(cache_t *cache, const void *ptr);
int cache_buf_isLocal(const cache_t *cache, const void *tile_ptr);
int cache_buf_distance(const cache_t *cache, const void *tile_ptr);
int cache_buf_age(const cache_t *cache, const void *tile_ptr);
