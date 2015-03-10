/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague_config.h>
#include <dague.h>
#include <string.h>
#include <stdlib.h>
#include "profiling.h"

void dague_profile_init_f08( int* ierr )
{
    *ierr = dague_profiling_init();
}

void dague_profile_fini_f08( int* ierr )
{
    *ierr = dague_profiling_fini();
}

void dague_profile_reset_f08( int* ierr )
{
    *ierr = dague_profiling_reset();
}

void dague_profile_dump_f08( int* ierr )
{
    *ierr = dague_profiling_dbp_dump();
}

void dague_profile_start_f08( const char *filename, const char *hr_info, int* ierr )
{
    *ierr = dague_profiling_dbp_start( filename, hr_info );
}

dague_thread_profiling_t*
dague_profile_thread_init_f08( size_t length, const char *id_name, int* ierr)
{
    dague_thread_profiling_t* tp = dague_profiling_thread_init(length, "%s", id_name);
    *ierr = (NULL == tp) ? -1 : 0;
    return tp;
}

void dague_profile_add_dictionary_keyword_f08(const char* key_name, const char* attributes,
                                              size_t info_length,
                                              const char* convertor_code,
                                              int* key_start, int* key_end, int* ierr )
{
    *ierr = dague_profiling_add_dictionary_keyword( key_name, attributes,
                                                    info_length,
                                                    convertor_code,
                                                    key_start, key_end );
}

void dague_profile_trace_f08( dague_thread_profiling_t** ctx, int key,
                              uint64_t event_id, uint32_t object_id,
                              void *info, int* ierr )
{
    *ierr = dague_profiling_trace(*ctx, key, event_id, object_id, info);
}
