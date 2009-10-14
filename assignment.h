#ifndef _assigment_h
#define _assigment_h

typedef struct assignment assignment_t;

#include "symbol.h"

struct assignment {
    symbol_t  *sym;
    int        value;
};

#endif
