/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _expr_h
#define _expr_h

typedef struct expr expr_t;

#include "symbol.h"
#include "assignment.h"

#define EXPR_OP_SYMB          1
#define EXPR_OP_CONST_INT     2

#define EXPR_OP_MIN_UNARY     10
#define EXPR_OP_UNARY_NOT     10
#define EXPR_OP_MAX_UNARY     10

#define EXPR_IS_UNARY(op)  ( ((op) >= EXPR_OP_MIN_UNARY) && ((op) <= EXPR_OP_MAX_UNARY) )

#define EXPR_OP_MIN_BINARY    20
#define EXPR_OP_BINARY_MOD    20
#define EXPR_OP_BINARY_EQUAL  21
#define EXPR_OP_BINARY_PLUS   22
#define EXPR_OP_BINARY_RANGE  23
#define EXPR_OP_BINARY_MINUS  24
#define EXPR_OP_BINARY_TIMES  25
#define EXPR_OP_MAX_BINARY    25

#define EXPR_IS_BINARY(op)  ( ((op) >= EXPR_OP_MIN_BINARY) && ((op) <= EXPR_OP_MAX_BINARY) )

struct expr {
    unsigned char op;
    union {
        struct {
            struct expr *op1;
            struct expr *op2;
        } binary;
        struct {
            struct expr *op1;
        } unary;
        symbol_t *var;
        int       const_int;
    } u;
};

#define bop1        u.binary.op1
#define bop2        u.binary.op2
#define uop1        u.unary.op1
#define var         u.var
#define const_int   u.const_int

#define EXPR_SUCCESS                        0
#define EXPR_FAILURE_SYMBOL_NOT_FOUND      -1
#define EXPR_FAILURE_UNKNOWN_OP            -2
#define EXPR_FAILURE_CANNOT_EVALUATE_RANGE -3
#define EXPR_FAILURE_UNKNOWN               -4

/**
 * Evaluates and returns the negation of an expression
 *
 * @param  [IN]  expr the expression to negate
 */
expr_t *negate_expr(expr_t *expr);

/**
 * Evaluates an expression in the current assignment context.
 *
 * @param  [IN]  expr the expression to evaluate
 * @param  [IN]  assignments the array of pairs (symbol, value) that define the evaluation context
 * @param  [IN]  nbassignments the size of assignments
 * @param  [OUT] res the evaluated value
 *
 * @return EXPR_SUCCESS in case of success. *res holds the evaluated value.
 * @return EXPR_FAILURE_* in case of error.
 */
int expr_eval( const expr_t *expr,
               const assignment_t *assignments,
               unsigned int nbassignments,
               int *res);

/**
 * Determine if the specified expression depend on the requested symbol or not.
 *
 * @param  [IN]  expr the expression to evaluate
 * @param  [IN]  the symbol to be looked for
 *
 * @return EXPR_SUCCESS if the expression depend on this particular symbol
 * @return EXPR_FAILURE_SYMBOL_NOT_FOUND if the expression do not depend on this symbol
 * @return EXPR_FAILURE_* in case of error.
 */
int expr_depend_on_symbol( const expr_t* expr,
                           const symbol_t* symbol );

/**
 * Determine if the specified expression depend on the requested symbol or not.
 *
 * @param  [IN]  expr the expression to evaluate
 * @param  [IN]  the callback to be called everytime a symbol is found in the expression.
 *               Cannot be NULL
 * @param  [IN]  callback data to be passed on the callback
 *
 * @return EXPR_SUCCESS if the expression depend on this particular symbol
 * @return EXPR_FAILURE_SYMBOL_NOT_FOUND if the expression do not depend on this symbol
 * @return EXPR_FAILURE_* in case of error.
 */
typedef int (expr_symbol_check_callback_t)(const symbol_t* symbol, void* data);
int expr_parse_symbols( const expr_t* expr,
                        expr_symbol_check_callback_t* callback,
                        void* data);

/**
 * Evaluates the minimum and maximum value of a range expression in the current assignment context.
 *
 * @param  [IN]  expr the expression to evaluate
 * @param  [IN]  assignments the array of pairs (symbol, value) that define the evaluation context
 * @param  [IN]  nbassignments the size of assignments
 * @param  [OUT] min the evaluated minimum value
 * @param  [OUT] max the evaluated maximum value
 *
 * @return EXPR_SUCCESS in case of success. *res holds the evaluated value.
 * @return EXPR_FAILURE_* in case of error.
 */
int expr_range_to_min_max( const expr_t *expr,
                           const assignment_t *assignments,
                           unsigned int nbassignments,
                           int *min,
                           int *max);


/**
 * Gives some comments on the first error encountered during the last call to expr_eval
 *
 * @return static string to the comment. Undefined if expr_eval has not been called previously.
 */
char *expr_error(void);

/**
 * Dumps an expression to the standard output (debugging purpose)
 *
 * @param [IN]  e the expression to dump
 */
void expr_dump(const expr_t *e);

/**
 * Creates a new expression from a variable name
 *
 * @param  name [IN]  the name of the variable
 * @return the new expression
 */
expr_t *expr_new_var(const symbol_t *name);

/**
 * Creates a new expression from a constant intenger
 *
 * @param  v [IN] value of the integer
 * @return the new expression
 */
expr_t *expr_new_int(int v);

/**
 * Creates a new binary expression from two others and an operand
 *
 * @param  op1 [IN]  first operand
 * @param  op  [IN] a character defining the operand in the alphabet
 *         [+-*%=.] where 
 *              = is the == comparison 
 *              . is the range operand.
 * @return the new expression
 */
expr_t *expr_new_binary(const expr_t *op1, char op, const expr_t *op2);

#endif
