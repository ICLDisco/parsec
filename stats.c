#include "dague_config.h"

#if defined(DAGuE_STATS)

#define DAGuE_STATS_C_DECLARE
#include "stats.h"
#undef _statsinternal_h
#undef stats_h
#undef DECLARE_STAT
#undef DECLARE_STATMAX
#undef DECLARE_STATACC
#undef DAGuE_STATS_C_DECLARE

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

#define DAGuE_STATS_C_DUMP
#include "stats.h"

    if( strcmp(filename, "-") ) {
        fclose(statfile);
    }
}

#endif /* defined (DAGuE_STATS) */
