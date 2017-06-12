/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef BINDTHREAD_H
#define BINDTHREAD_H

/** @addtogroup parsec_internal_binding
 *  @{
 */

int parsec_bindthread(int cpu, int ht);

#if defined(PARSEC_HAVE_HWLOC)
#include <hwloc.h>
int parsec_bindthread_mask(hwloc_cpuset_t cpuset);
#endif

/** @} */

#endif /* BINDTHREAD_H */
