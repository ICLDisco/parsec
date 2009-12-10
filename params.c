/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dplasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* Have to love that ... */
extern char *strdup(const char *);

void param_dump(const param_t *p, const char *prefix)
{
    int i, length;
    char *pref2 = (char*)malloc(strlen(prefix)+8);

    length = printf("%s%s %s%s ", 
                    prefix, (p->name == NULL) ? "" : p->name, 
                    (p->sym_type & SYM_IN)  ? "IN"  : "  ",
                    (p->sym_type & SYM_OUT) ? "OUT" : "   ");

    if( NULL != p->dep_in[0] ) {
        dep_dump(p->dep_in[0], "<- " );
        for( i = sprintf( pref2, "%s", prefix ); i < length; pref2[i] = ' ', i++ );
        sprintf( pref2 + length, "<- " );
    }
    for(i = 1; NULL != p->dep_in[i] && i < MAX_DEP_IN_COUNT; i++) {
        dep_dump( p->dep_in[i], pref2 );
    }

    sprintf( pref2 + length, "-> " );
    for(i = 0; i < MAX_DEP_OUT_COUNT && NULL != p->dep_out[i]; i++) {
        dep_dump( p->dep_out[i], pref2 );
    }
}

param_t* dplasma_find_or_create_param(dplasma_t* function, char* param_name)
{
    param_t* param;
    int i;

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->params[i]); i++ ) {
        param = function->params[i];
        if( 0 == strcmp(param->name, param_name) ) {
            return param;
        }
    }
    if( i == MAX_PARAM_COUNT ) {
        fprintf( stderr, "Too many parameters for function %s (stopped at %s)\n",
                 function->name, param_name );
        return NULL;
    }
    param = (param_t*)calloc(1, sizeof(param_t));
    param->name = strdup((const char*)param_name);
    param->function = function;
    param->param_mask = (1 << i);
    function->params[i] = param;
    return param;
}
