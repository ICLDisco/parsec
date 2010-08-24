/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dep.h"

void dep_dump(const dep_t *d, const struct dague_object *dague_object, const char *prefix)
{
    int i;
    printf("%s", prefix);
    if( NULL != d->cond ) {
        printf("if");
        expr_dump(stdout, dague_object, d->cond);
        printf(" then ");
    }
    printf( "%s %s(",
            (d->param == NULL ? "" : d->param->name), d->dague->name);
    for(i = 0; i < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i]; i++) {
        expr_dump(stdout, dague_object, d->call_params[i]);
        if( i+1 < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i+1] ) {
            printf(", ");
        } 
    }
    printf(")\n");
}
