#ifndef _symbol_h
#define _symbol_h

#include "expr.h"

typedef struct symbol {
    char   *name;
    expr_t *min;
    expr_t *max;
} symbol_t;

#endif
