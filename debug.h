/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEBUG_H_HAS_BEEN_INCLUDED
#define DEBUG_H_HAS_BEEN_INCLUDED

#include "dplasma_config.h"

#ifdef DPLASMA_DEBUG

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#   ifdef USE_MPI
/* only one printf to avoid line breaks in the middle */

#include <stdarg.h>
static inline char* arprintf(const char* fmt, ...)
{
    char* txt;
    va_list args;
    
    va_start(args, fmt);
    vasprintf(&txt, fmt, args);
    va_end(args);
    return txt;
}

#include <mpi.h>

#define DEBUG(ARG)  do { \
    int __debug_rank; \
    char* __debug_str; \
    MPI_Comm_rank(MPI_COMM_WORLD, &__debug_rank); \
    __debug_str = arprintf ARG ; \
    fprintf(stderr, "[%d]\t%s", __debug_rank, __debug_str); \
    free(__debug_str); \
} while(0)

#   else /* USE_MPI */

#define DEBUG(ARG) printf ARG

#   endif /* USE_MPI */

#else /* DPLASMA_DEBUG */

#define DEBUG(ARG)

#endif /* DPLASMA_DEBUG */

#endif /* DEBUG_H_HAS_BEEN_INCLUDED */

