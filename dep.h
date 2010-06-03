/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _dep_h
#define _dep_h

typedef struct dep dep_t;

#include "DAGuE.h"
#include "expr.h"

#define MAX_CALL_PARAM_COUNT    MAX_PARAM_COUNT

struct dep {
    const expr_t*    cond;
    const DAGuE_t*   DAGuE;
    const expr_t*    call_params[MAX_CALL_PARAM_COUNT];
    const param_t*   param;
    void*            type;
};

void dep_dump(const dep_t *d, const char *prefix);

#endif
