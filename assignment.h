/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _assigment_h
#define _assigment_h

typedef struct assignment assignment_t;

#include "symbol.h"

struct assignment {
    const symbol_t*  sym;
    int        value;
    int        min;
    int        max;
};

#define DAGUE_ASSIGN_FOUND   0
#define DAGUE_ASSIGN_ADDED   1
#define DAGUE_ASSIGN_ERROR  -1

/**
 * Returns the assignment pointer if a symbol with the specified name is
 * present in the list of assignments.
 *
 * @param [IN]  The name of the symbol to be found. Cannot be NULL.
 * @param [IN]  A list of valid assignments. An assignment with a symbol
 *              set to NULL denote the end of the list.
 * @param [IN]  The maximum number of assignments in the list.
 * @param [OUT] The pointer to the assignment related to this symbol if
 *              the symbol was found. Otherwise this value is not modified.
 *
 * @return DAGUE_ASSIGN_FOUND if the symbol is correctly resolved and the
 *                              return value has a meaning.
 * @return DAGUE_ASSIGN_ERROR if the symbol is not in the assignments list. 
 */
int dague_find_assignment( const char* name,
                             const assignment_t* context,
                             unsigned int context_size,
                             assignment_t** where);

/**
 * Returns the assignment pointer if the requested symbol is in the assignments list
 * and add it to the assignment list if not.
 *
 * @param [IN]  The symbol to be analyzed. Cannot be NULL.
 * @param [IN]  A list of valid assignments. An assignment with a symbol
 *              set to NULL denote the end of the list. If the symbol is
 *              not in the list it will be added.
 * @param [IN]  The maximum number of assignments in the list.
 * @param [OUT] The pointer to the assignment related to this symbol if
 *              the symbol was found. Otherwise this value is not modified.
 *
 * @return DAGUE_ASSIGN_FOUND if the symbol was already in the assignment list.
 * @return DAGUE_ASSIGN_ADDED is the symbol was not initially in the assignments
 *                              list, and it was succesfully added.
 * @return DAGUE_ASSIGN_ERROR if something bad happened. The assignments list
 *                              is untouched.
 */
int dague_add_assignment( const symbol_t* symbol,
                            assignment_t* context,
                            unsigned int context_size,
                            assignment_t** where );

#endif
