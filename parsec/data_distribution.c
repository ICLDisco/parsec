/*
 * Copyright (c) 2017      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/data_distribution.h"

#if defined(PARSEC_HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* defined(PARSEC_HAVE_STDARG_H) */
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* defined(PARSEC_HAVE_UNISTD_H) */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void
parsec_ddesc_init(parsec_ddesc_t *d,
                  int nodes, int myrank )
{
    memset( d, 0, sizeof(parsec_ddesc_t) );

    d->nodes  = nodes;
    d->myrank = myrank;
    d->tile_h_table = NULL;
    d->memory_registration_status = MEMORY_STATUS_UNREGISTERED;
}

void
parsec_ddesc_destroy(parsec_ddesc_t *d)
{
#if defined(PARSEC_PROF_TRACE)
    if( NULL != d->key_dim ) free(d->key_dim);
    d->key_dim = NULL;
#endif
    if( NULL != d->key_base ) free(d->key_base);
    d->key_base = NULL;
}

#if defined(PARSEC_PROF_TRACE)
#include "parsec/profiling.h"

void parsec_ddesc_set_key( parsec_ddesc_t* d, char* name)
{
    char dim[strlen(name) + strlen( (d)->key_dim ) + 4];
    (d)->key_base = strdup(name);
    sprintf(dim, "%s%s", name, (d)->key_dim);
    parsec_profiling_add_information( "DIMENSION", dim );
}
#endif  /* defined(PARSEC_PROF_TRACE) */
