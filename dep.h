/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dep_h
#define _dep_h

typedef struct dep dep_t;

#include "expr.h"
#include "dplasma.h"

#define MAX_CALL_PARAM_COUNT    MAX_PARAM_COUNT

struct dep {
    expr_t    *cond;
    dplasma_t *dplasma;
    expr_t    *call_params[MAX_CALL_PARAM_COUNT];
    char      *sym_name;
};

void dep_dump(const dep_t *d, const char *prefix);
char *dump_c_dep(FILE *out, const dep_t *d, const char *prefix);

#endif
