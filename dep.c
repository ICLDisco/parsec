/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dep.h"

void dep_dump(const dep_t *d, const char *prefix)
{
    int i;
    printf("%s", prefix);
    if( NULL != d->cond ) {
        printf("if");
        expr_dump(stdout, d->cond);
        printf(" then ");
    }
    printf( "%s %s(",
            (d->param == NULL ? "" : d->param->name), d->dplasma->name);
    for(i = 0; i < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i]; i++) {
        expr_dump(stdout, d->call_params[i]);
        if( i+1 < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i+1] ) {
            printf(", ");
        } 
    }
    printf(")\n");
}
