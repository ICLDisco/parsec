#include "dplasma_config.h"

#if defined(DPLASMA_STATS)

#define DPLASMA_STATS_C_DECLARE
#include "stats.h"
#undef _statsinternal_h
#undef stats_h
#undef DECLARE_STAT
#undef DPLASMA_STATS_C_DECLARE

#include <stdio.h>
#include <string.h>
#include <errno.h>

void dplasma_stats_dump(char *filename, char *prefix)
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

#define DPLASMA_STATS_C_DUMP
#include "stats.h"

    if( strcmp(filename, "-") ) {
        fclose(statfile);
    }
}

#endif /* defined (DPLASMA_STATS) */
