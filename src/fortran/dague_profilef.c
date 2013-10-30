/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague_config.h>
#include <dague.h>
#include <string.h>
#include <stdlib.h>
#include "profiling.h"

void dague_profile_init_f08( const char *hdr_id, int* ierr )
{
    *ierr = dague_profiling_init("%s", hdr_id);
}

void dague_profile_fini_f08( int* ierr )
{
    *ierr = dague_profiling_fini();
}

void dague_profile_reset_f08( int* ierr )
{
    *ierr = dague_profiling_reset();
}

void dague_profile_dump_f08( const char* filename, int* ierr )
{
    *ierr = dague_profiling_dump_dbp(filename);
}

dague_thread_profiling_t*
dague_profile_thread_init_f08( size_t length, const char *id_name)
{
    return dague_profiling_thread_init(length, "%s", id_name);
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
