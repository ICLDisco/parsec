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
