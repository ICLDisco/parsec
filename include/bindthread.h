/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef BINDTHREAD_H
#define BINDTHREAD_H

int dague_bindthread(int cpu, int ht);

#if defined(HAVE_HWLOC)
#include <hwloc.h>
int dague_bindthread_mask(hwloc_cpuset_t cpuset);
#endif

#endif /* BINDTHREAD_H */
