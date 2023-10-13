/*
 * Copyright (c) 2013-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/parsec_config.h"
#include "parsec/runtime.h"
#include <string.h>
#include <stdlib.h>
#include "parsec/profiling.h"

parsec_profiling_stream_t*
parsec_profile_thread_init_f08( size_t length, const char *id_name, int* ierr)
{
    parsec_profiling_stream_t* tp = parsec_profiling_stream_init(length, "%s", id_name);
    *ierr = (NULL == tp) ? -1 : 0;
    return tp;
}

void parsec_profile_add_dictionary_keyword_f08(const char* key_name, const char* attributes,
                                              size_t info_length,
                                              const char* convertor_code,
                                              int* key_start, int* key_end, int* ierr )
{
    *ierr = parsec_profiling_add_dictionary_keyword( key_name, attributes,
                                                    info_length,
                                                    convertor_code,
                                                    key_start, key_end );
}
