/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "params.h"

void param_dump(const param_t *p, const char *prefix)
{
    int i;
    char *pref2 = (char*)malloc(strlen(prefix)+8);
    pref2[0] = '\0';

    printf("%s%s %s%s ", 
           prefix, p->sym_name==NULL?"":p->sym_name, 
           (p->sym_type & SYM_IN)  ? "IN"  : "  ",
           (p->sym_type & SYM_OUT) ? "OUT" : "   ");
    fflush(stdout);

    for(i = 0; NULL != p->dep_in[i] && i < MAX_DEP_IN_COUNT; i++) {
        dep_dump( p->dep_in[i], pref2 );
        sprintf(pref2, "%s       ", prefix);
    }
    for(i = 0; i < MAX_DEP_OUT_COUNT && NULL != p->dep_out[i]; i++) {
        dep_dump( p->dep_out[i], pref2 );
        sprintf(pref2, "%s       ", prefix);
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
        l += snprintf(param + l, 4096-l, "static param_t param%d = { .sym_name = \"%s\", .sym_type = %d,\n     .dep_in  = {", param_idx, p->sym_name, p->sym_type);
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
