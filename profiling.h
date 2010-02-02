/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dplasma_profiling_h
#define _dplasma_profiling_h

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

#include "execution_unit.h"

/**
 * Initialize the circular trace with the specified length. If threads are
 * enabled then the trace is per thread and each one of them is supposed to
 * call the tracing initialization function.
 */
int dplasma_profiling_init( dplasma_context_t* context, size_t length );

/**
 * Release all resources for the tracing. If threads are enabled only
 * the resources related to this thread are released.
 */
int dplasma_profiling_fini( dplasma_context_t* context );

int dplasma_profiling_add_dictionary_keyword( const char*, const char* attributes,
                                              int* key_start, int* key_end );
int dplasma_profiling_del_dictionary_keyword( int key );

int dplasma_profiling_reset( dplasma_execution_unit_t* context );

int dplasma_profiling_trace( dplasma_execution_unit_t* context, int key, unsigned long id );

int dplasma_profiling_dump_svg( dplasma_context_t* context, const char* filename );

#endif  /* _dplasma_profiling_h */
