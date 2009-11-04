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
#define DPLASMA_SYMBOL_IS_GLOBAL      0x0001
/* This symbol doesn't depend on any other local symbols. However,
 * it can depend on global symbols */
#define DPLASMA_SYMBOL_IS_STANDALONE  0x0002

struct symbol {
    uint32_t flags;
    const char*   name;
    const expr_t* min;
    const expr_t* max;
};

/**
 * Dump the specified symbol.
 */
void symbol_dump(const symbol_t *s, const char *prefix);
char *dump_c_symbol(FILE *out, const symbol_t *s, const char *prefix);

/**
 * helper for dumping the c structure representing the dplasma object
 */
int symbol_c_index_lookup( const symbol_t *symbol );

/**
 * Dump all globally defined symbols.
 */
void symbol_dump_all( const char* prefix );

/**
 * Search for a global symbol.
 */
const symbol_t* dplasma_search_global_symbol( const char* name );

/**
 * Return 1 if the symbol is global.
 */
static inline int dplasma_symbol_is_global( const symbol_t* symbol )
{
    return (symbol->flags & DPLASMA_SYMBOL_IS_GLOBAL ? 1 : 0);
}

/**
 * Add a global symbol.
 */
int dplasma_add_global_symbol( const char* name, const expr_t* expr );

/**
 * Return 0 if the symbol is standalone, i.e. it doesn't depend on any
 * local symbols only on constants or global symbols.
 *
 * @param [IN]  The symbol to be analyzed. Cannot be NULL.
 *
 * @return  0 if the symbol is standalone.
 * @return -1 otherwise (no specific error returned).
 */
int dplasma_symbol_is_standalone( const symbol_t* symbol );

/**
 * Return the first acceptable value for a specific symbol. As a result the symbol
 * will be either added or updated on the assignment array.
 *
 * @param [IN]  The symbol to be analyzed. Cannot be NULL.
 * @param [IN]  A NULL terminated array of predicates, eventually some regarding the
 *              symbol to be analyzed. Can be NULL if no predicated
 *              are imposed.
 * @param [IN]  The list of symbols and their current values in the
 *              current execution context.
 * @param [OUT] The first acceptable value for this symbol.
 *
 * @return  0 if the symbol was correctly resolved and the return value
 *            has a meaning.
 * @return -1 if something bad happened and the returned value cannot
 *            be used.
 */
int dplasma_symbol_get_first_value( const symbol_t* symbol,
                                    const expr_t** predicates,
                                    assignment_t* local_context,
                                    int* pvalue );
/**
 * Return the last acceptable value for a specific symbol. As a result the symbol
 * will be either added or updated on the assignment array.
 *
 * @param [IN]  The symbol to be analyzed. Cannot be NULL.
 * @param [IN]  A NULL terminated array of predicates, eventually some regarding the
 *              symbol to be analyzed. Can be NULL if no predicated
 *              are imposed.
 * @param [IN]  The list of symbols and their current values in the
 *              current execution context.
 * @param [OUT] The last acceptable value for this symbol.
 *
 * @return  0 if the symbol was correctly resolved and the return value
 *            has a meaning.
 * @return -1 if something bad happened and the returned value cannot
 *            be used.
 */
int dplasma_symbol_get_last_value( const symbol_t* symbol,
                                   const expr_t** predicates,
                                   assignment_t* local_context,
                                   int* pvalue );
/**
 * Return the next acceptable value for a specific symbol. As a result the symbol
 * will be either added or updated on the assignment array.
 *
 * @param [IN]  The symbol to be analyzed. Cannot be NULL.
 * @param [IN]  A NULL terminated array of predicates, eventually some regarding the
 *              symbol to be analyzed. Can be NULL if no predicated
 *              are imposed.
 * @param [IN]  The list of symbols and their current values in the
 *              current execution context.
 * @param [INOUT] On input it contains the current value of the symbol,
 *                while on output it contains the next acceptable value.
 *
 * @return  0 if the symbol was correctly resolved and the return value
 *            has a meaning.
 * @return -1 if something bad happened and the returned value cannot
 *            be used.
 */
int dplasma_symbol_get_next_value( const symbol_t* symbol,
                                   const expr_t** predicates,
                                   assignment_t* local_context,
                                   int* pvalue );

/**
 * Return the absolute minimal value for a specific symbol.
 *
 * @param [IN]  The symbol to be analyzed. Cannot be NULL.
 * @param [OUT] The absolute minimal acceptable value for this symbol.
 *
 * @return  0 if the symbol was correctly resolved and the return value
 *            has a meaning.
 * @return -1 if something bad happened and the returned value cannot
 *            be used.
 */
int dplasma_symbol_get_absolute_minimum_value( const symbol_t* symbol,
                                               int* pvalue );

/**
 * Return the absolute maximal value for a specific symbol.
 *
 * @param [IN]  The symbol to be analyzed. Cannot be NULL.
 * @param [IN]  A NULL terminated array of symbols from the same dplama_t object, which
 *              might create dependencies with the analyzed one. In the case this list is NULL,
 *              we suppose the upper layer already knows there are no dependencies.
 * @param [OUT] The absolute maximal acceptable value for this symbol.
 *
 * @return  0 if the symbol was correctly resolved and the return value
 *            has a meaning.
 * @return -1 if something bad happened and the returned value cannot
 *            be used.
 */
int dplasma_symbol_get_absolute_maximum_value( const symbol_t* symbol,
                                               int* pvalue );

#endif
