#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
// Forward declarations

void dplasma_hwloc_init_cache(int npu, int level, int npu_per_cache, int cache_size, int tile_size);
void dplasma_hwloc_insert_buffer(void *array_ptr, int bufSize, int myPUID);
int dplasma_hwloc_isLocal(void *array_ptr, int cacheLevel, int myPUID);
