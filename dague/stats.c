/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"

#if defined(DAGUE_STATS)

#define DAGUE_STATS_C_DECLARE
#include "stats.h"
#undef _statsinternal_h
#undef stats_h
#undef DECLARE_STAT
#undef DECLARE_STATMAX
#undef DECLARE_STATACC
#undef DAGUE_STATS_C_DECLARE

#include <stdio.h>
#include <string.h>
#include <errno.h>

void dague_stats_dump(char *filename, char *prefix)
{
    FILE *statfile;

    if( strcmp(filename, "-") ) {
        statfile = fopen(filename, "w");
        if(statfile == NULL) {
            fprintf(stderr, "unable to open %s: %s\n", filename, strerror(errno));
            return;
        }
    } else {
        statfile = stdout;
    }

#define DAGUE_STATS_C_DUMP
#include "stats.h"

    if( strcmp(filename, "-") ) {
        fclose(statfile);
    }
}

#endif /* defined (DAGUE_STATS) */
