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

typedef struct dumped_param_list {
    const param_t *param;
    unsigned int   idx;
    char          *param_name;
    struct dumped_param_list *next;
} dumped_param_list_t;

char *dump_c_param(FILE *out, const param_t *p, char *init_func_body, int init_func_body_size, int dump_it)
{
    static unsigned int param_idx = 0;
    static dumped_param_list_t *dumped_params = NULL;
    static char name[64];
    dumped_param_list_t *dumped;
    char param[4096];
    int  l = 0;
    int i;
    char *dep_name;
    unsigned int my_idx;

    if( p == NULL ) {
        sprintf(name, "NULL");
    } else {
        for(dumped = dumped_params; dumped != NULL; dumped = dumped->next) {
            if( dumped->param == p ) {
                if( !dump_it ) {
                    return dumped->param_name;
                } else {
                    my_idx = dumped->idx;
                    break;
                }
            }
        }

        if( dumped == NULL ) {
            my_idx = param_idx++;
            dumped = (dumped_param_list_t*)calloc(1, sizeof(dumped_param_list_t));
            dumped->param = p;
            dumped->idx = my_idx;
            asprintf(&dumped->param_name, "&param%d", my_idx);
            dumped->next = dumped_params;
            dumped_params = dumped;
            if( !dump_it ) {
                return dumped->param_name;
            }
        }

        l += snprintf(param + l, 4096-l, 
                      "static param_t param%d = { .name = \"%s\", .sym_type = %d, .param_mask = 0x%02x,\n"
                      "     .dep_in  = {", my_idx, p->name, p->sym_type, p->param_mask);
        for(i = 0; i < MAX_DEP_IN_COUNT; i++) {
            dep_name = dump_c_dep(out, p->dep_in[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, 4096-l, "%s%s", dep_name, i < MAX_DEP_IN_COUNT-1 ? ", " : "},\n"
                          "     .dep_out = {");
        }
        for(i = 0; i < MAX_DEP_OUT_COUNT; i++) {
            dep_name = dump_c_dep(out, p->dep_out[i], init_func_body, init_func_body_size);
            l += snprintf(param + l, 4096-l, "%s%s", dep_name, i < MAX_DEP_OUT_COUNT-1 ? ", " : "} };\n");
        }
        fprintf(out, "%s", param);
        snprintf(name, 64, "&param%d", my_idx);
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
    param->name = strdup((const char*)param_name);
    param->function = function;
    param->param_mask = (1 << i);
    function->params[i] = param;
    return param;
}
