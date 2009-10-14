#ifndef _dplasma_h
#define _dplasma_h

#include "symbol.h"

#define MAX_LOCAL_COUNT  3


typedef struct dplasma {
    char      *name;
    symbol_t  *locals[MAX_LOCAL_COUNT];
    
} dplasma_t;

#endif
