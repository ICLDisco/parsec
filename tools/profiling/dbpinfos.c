/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#undef PARSEC_HAVE_MPI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdarg.h>

#include "parsec/profiling.h"
#include "parsec/parsec_binary_profile.h"
#include "dbpreader.h"

int main(int argc, char *argv[])
{
    dbp_multifile_reader_t *dbp;
    int ifd, i;
    dbp_file_t *file;
    dbp_info_t *info;
    char *key;
    char *value;

    dbp = dbp_reader_open_files(argc-1, argv+1);

    for(ifd = 0; ifd < dbp_reader_nb_files(dbp); ifd++) {
        file = dbp_reader_get_file(dbp, ifd);
        printf("==================== %s ====================\n", dbp_file_get_name(file));
        for(i = 0; i < dbp_file_nb_infos(file); i++) {
            info = dbp_file_get_info(file, i);
            key = dbp_info_get_key(info);
            value = dbp_info_get_value(info);
            if( strlen(value) + strlen(key) + 4 < 80 ) {
                printf("  %s: %s\n", key, value);
            } else {
                printf("  %s:\n", key);
                printf("%s\n\n", value);
            }
        }
    }

    dbp_reader_close_files(dbp);
    free(dbp);

    return 0;
}
