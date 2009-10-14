#ifndef _dplasma_h
#define _dplasma_h

typedef struct dplasma dplasma_t;

#define MAX_LOCAL_COUNT  3
#define MAX_PRED_COUNT   3
#define MAX_PARAM_COUNT  3

#include "symbol.h"
#include "expr.h"
#include "params.h"

struct dplasma {
    char      *name;
    symbol_t  *locals[MAX_LOCAL_COUNT];
    expr_t    *preds[MAX_PRED_COUNT];
    param_t   *params[MAX_PARAM_COUNT];
    char      *body;
};

#endif
