/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

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
        expr_dump(d->cond);
        printf(" then ");
    }
    printf( "%s%s%s(",
            (d->sym_name == NULL ? "" : d->sym_name),
            (d->sym_name != NULL ? " " : ""), d->dplasma->name);
    for(i = 0; i < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i]; i++) {
        expr_dump(d->call_params[i]);
        if( i+1 < MAX_CALL_PARAM_COUNT && NULL != d->call_params[i+1] ) {
            printf(", ");
        } 
    }
    printf(")\n");
}

char *dump_c_dep(FILE *out, const dep_t *d, char *init_func_body, int init_func_body_size)
{
    static unsigned int dep_idx = 0;
    static char name[64];
    int i;
    
    if( d == NULL ) {
        sprintf(name, "NULL");
    } else {
        char whole[4096];
        int p = 0;

        sprintf(name, "&dep%d", dep_idx);

        p += snprintf(whole + p, 4096-p, 
                      "static dep_t dep%d = { .cond = %s, .dplasma = NULL,\n"
                      "                       .call_params = {",
                      dep_idx, dump_c_expression(out, d->cond, init_func_body, init_func_body_size));
        i = snprintf(init_func_body + strlen(init_func_body), init_func_body_size - strlen(init_func_body),
                     "  dep%d.dplasma = &dplasma_array[%d];\n", dep_idx, dplasma_dplasma_index( d->dplasma ));
        if(i + strlen(init_func_body) >= init_func_body_size ) {
            fprintf(stderr, "Ah! Thomas told us so: %d is too short for the initialization body function\n",
                    init_func_body_size);
        }
        for(i = 0 ; i < MAX_CALL_PARAM_COUNT; i++) {
            p += snprintf(whole + p, 4096-p, "%s%s", dump_c_expression(out, d->call_params[i], init_func_body, init_func_body_size), 
                          i < MAX_CALL_PARAM_COUNT-1 ? ", " : "},\n");
        }
        p += snprintf(whole + p, 4096-p, "                       .sym_name = \"%s\" };\n", d->sym_name);
        fprintf(out, "%s", whole);
        dep_idx++;
    }
     
   return name;
}
