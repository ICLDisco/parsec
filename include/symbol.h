/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _symbol_h
#define _symbol_h

typedef struct symbol symbol_t;

#include "expr.h"
#include "assignment.h"
#include <stdint.h>

/* This symbol is a global one. */
#define DAGUE_SYMBOL_IS_GLOBAL      0x0001
/* This symbol doesn't depend on any other local symbols. However,
 * it can depend on global symbols */
#define DAGUE_SYMBOL_IS_STANDALONE  0x0002

struct symbol {
    uint32_t flags;
    const char*   name;
    const char*   type;
    const expr_t* min;
    const expr_t* max;
};

/**
 * Return 1 if the symbol is global.
 */
static inline int dague_symbol_is_global( const symbol_t* symbol )
{
    return (symbol->flags & DAGUE_SYMBOL_IS_GLOBAL ? 1 : 0);
}

#endif
