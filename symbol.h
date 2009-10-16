#ifndef _symbol_h
#define _symbol_h

typedef struct symbol symbol_t;

#include "expr.h"

struct symbol {
    const char*   name;
    const expr_t* min;
    const expr_t* max;
};

/**
 * Dump the specified symbol.
 */
void symbol_dump(const symbol_t *s, const char *prefix);

/**
 * Dump all globally defined symbols.
 */
void symbol_dump_all( const char* prefix );

/**
 * Search for a global symbol.
 */
const symbol_t* dplasma_search_global_symbol( const char* name );

/**
 * Add a global symbol.
 */
int dplasma_add_global_symbol( const char* name, const expr_t* expr );

#endif
