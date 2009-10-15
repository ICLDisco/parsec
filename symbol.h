#ifndef _symbol_h
#define _symbol_h

typedef struct symbol symbol_t;

#include "expr.h"

struct symbol {
    char   *name;
    expr_t *min;
    expr_t *max;
};

void symbol_dump(const symbol_t *s, const char *prefix);

#endif
