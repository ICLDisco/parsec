#ifndef _walk_utils_h
#define _walk_utils_h
/*
 * Copyright (c) 2016-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "tree_dist.h"

typedef void (*walk_fn_t)(tree_dist_t *, node_t *, int , int, void *);

#endif

