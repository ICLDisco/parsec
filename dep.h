#ifndef _dep_h
#define _dep_h

#include "expr.h"
#include "dplasma.h"

#define MAX_CALL_PARAM_COUNT    MAX_PARAM_COUNT

typedef struct {
    expr_t    *cond;
    dplasma_t *t;
    expr_t    *call_params[MAX_CALL_PARAM_COUNT];
    char      *sym_name;
} dep_t;

#endif
