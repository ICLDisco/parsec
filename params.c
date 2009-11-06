/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "dplasma.h"

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

char *dump_c_param(FILE *out, const param_t *p, char *init_func_body, int init_func_body_size)
{
    static unsigned int param_idx = 0;
    static char name[64];
    char param[4096];
    int  l = 0;
    int i;

    if( p == NULL ) {
        sprintf(name, "NULL");
    } else {
        sprintf(name, "&param%d", param_idx);
        l += snprintf(param + l, 4096-l, "static param_t param%d = { .name = \"%s\", .sym_type = %d,\n     .dep_in  = {", param_idx, p->name, p->sym_type);
        for(i = 0; i < MAX_DEP_IN_COUNT; i++) {
            l += snprintf(param + l, 4096-l, "%s%s", dump_c_dep(out, p->dep_in[i], init_func_body, init_func_body_size), i < MAX_DEP_IN_COUNT-1 ? ", " : "},\n     .dep_out = {");
        }
        for(i = 0; i < MAX_DEP_OUT_COUNT; i++) {
            l += snprintf(param + l, 4096-l, "%s%s", dump_c_dep(out, p->dep_out[i], init_func_body, init_func_body_size), i < MAX_DEP_OUT_COUNT-1 ? ", " : "} };\n");
        }
        fprintf(out, "%s", param);
        param_idx++;
    }

    return name;
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
    param->name = strdup(param_name);
    param->function = function;
    param->param_mask = (1 << i);
    function->params[i] = param;
    return param;
}
