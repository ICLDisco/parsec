#include <stdio.h>
#include "symbol.h"

void symbol_dump(const symbol_t *s, const char *prefix)
{
    printf("%s%s = [", prefix, s->name);
    expr_dump(s->min);
    printf(" .. ");
    expr_dump(s->max);
    printf("]\n");
}
