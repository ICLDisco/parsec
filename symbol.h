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
#endif
