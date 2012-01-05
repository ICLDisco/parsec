#ifndef BINDTHREAD_H
#define BINDTHREAD_H

int dague_bindthread(int cpu);

#if defined(HAVE_HWLOC)
#include <hwloc.h>
int dague_bindthread_mask(hwloc_cpuset_t cpuset);
#endif

#endif /* BINDTHREAD_H */
