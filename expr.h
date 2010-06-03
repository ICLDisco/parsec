/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _expr_h
#define _expr_h

typedef struct expr expr_t;

#include <stdio.h>

#include "symbol.h"
#include "assignment.h"

#define EXPR_OP_SYMB               1
#define EXPR_OP_CONST_INT          2

#define EXPR_OP_MIN_UNARY         10
#define EXPR_OP_UNARY_NOT         10
#define EXPR_OP_MAX_UNARY         10

#define EXPR_IS_UNARY(op)  ( ((op) >= EXPR_OP_MIN_UNARY) && ((op) <= EXPR_OP_MAX_UNARY) )

#define EXPR_OP_MIN_BINARY            20
#define EXPR_OP_BINARY_MOD            20
#define EXPR_OP_BINARY_EQUAL          21
#define EXPR_OP_BINARY_NOT_EQUAL      22
#define EXPR_OP_BINARY_PLUS           23
#define EXPR_OP_BINARY_RANGE          24
#define EXPR_OP_BINARY_MINUS          25
#define EXPR_OP_BINARY_TIMES          26
#define EXPR_OP_BINARY_DIV            27
#define EXPR_OP_BINARY_OR             28
#define EXPR_OP_BINARY_AND            29
#define EXPR_OP_BINARY_XOR            30
#define EXPR_OP_BINARY_LESS           31
#define EXPR_OP_BINARY_LESS_OR_EQUAL  32
#define EXPR_OP_BINARY_MORE           33
#define EXPR_OP_BINARY_MORE_OR_EQUAL  34
#define EXPR_OP_BINARY_SHL            35
#define EXPR_OP_MAX_BINARY            35

#define EXPR_IS_BINARY(op)  ( ((op) >= EXPR_OP_MIN_BINARY) && ((op) <= EXPR_OP_MAX_BINARY) )

#define EXPR_OP_MIN_TERTIAR           70
#define EXPR_OP_CONDITIONAL           70
#define EXPR_OP_MAX_TERTIAR           70
#define EXPR_IS_TERTIAR(op)  ( ((op) >= EXPR_OP_MIN_TERTIAR) && ((op) <= EXPR_OP_MAX_TERTIAR) )

#define EXPR_OP_INLINE                100
#define EXPR_IS_INLINE(op)  ( (op) == EXPR_OP_INLINE )

/**
 * Flags to be used with the expressions to speed-up their evaluation.
 */
#define EXPR_FLAG_CONSTANT   0x01

struct dague_object;
typedef int (*expr_op_inline_func_t)(const struct dague_object *__dague_object_parent, const assignment_t *assignments);

struct expr {
    unsigned char op;
    unsigned char flags;
    int           value;  /* value for the constant expressions or for the
                           * expressions that can be evaluated to a
                           * constant.
                           */
    union {
        struct {
            const struct expr *cond;
            const struct expr *op1;
            const struct expr *op2;
        } tertiar;
        struct {
            const struct expr *op1;
            const struct expr *op2;
        } binary;
        struct {
            const struct expr *op1;
        } unary;
        const symbol_t *var;
        expr_op_inline_func_t inline_func;
    } u_expr;
};

#define tcond       u_expr.tertiar.cond
#define top1        u_expr.tertiar.op1
#define top2        u_expr.tertiar.op2
#define bop1        u_expr.binary.op1
#define bop2        u_expr.binary.op2
#define uop1        u_expr.unary.op1
#define variable    u_expr.var
#define const_int   u_expr.const_int
#define inline_func u_expr.inline_func

#define EXPR_SUCCESS                        0
#define EXPR_FAILURE_SYMBOL_NOT_FOUND      -1
#define EXPR_FAILURE_UNKNOWN_OP            -2
#define EXPR_FAILURE_CANNOT_EVALUATE_RANGE -3
#define EXPR_FAILURE_UNKNOWN               -4

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
 * @return EXPR_SUCCESS in case of success.
 * @return EXPR_FAILURE_* in case of error.
 */
int expr_range_to_min_max( const expr_t *expr,
                           const assignment_t *assignments,
                           unsigned int nbassignments,
                           int *min,
                           int *max);

/**
 * Evaluates the absolute minimum and maximum value of an expression/
 *
 * @param  [IN]  expr the expression to be evaluated
 * @param  [OUT] pmin pointer to the evaluated minimum value
 * @param  [OUT] pmax pointer to the evaluated maximum value
 *
 * @return EXPR_SUCCESS in case of success.
 * @return EXPR_FAILURE_* in case of error.
 */
int expr_absolute_range(const expr_t* expr,
                        int* pmin, int* pmax);

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
void expr_dump(FILE *out, const expr_t *e);

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
 * Create a new unary expression based on the operand and expression.
 *
 * @param  [IN]  op a character defining the operand
 * @param  [IN]  expr the expression to negate
 */
expr_t *expr_new_unary(char op, expr_t *expr);

/**
 * Creates a new binary expression from two others and an operand
 *
 * @param  op1 [IN]  first operand
 * @param  op2 [IN]  first operand
 * @param  op  [IN] a character defining the operand in the alphabet
 *         [+-*%=.] where 
 *              = is the == comparison 
 *              . is the range operand.
 * @return the new expression
 */
expr_t *expr_new_binary(const expr_t *op1, char op, const expr_t *op2);

/**
 * Creates a new tertiar expression from two others and a condition.
 * If the condition is true the expression evaluate as the op1,
 * otherwise it evaluate as op2.
 *
 * @param  cond [IN]  first operand
 * @param  op1  [IN]  first operand
 * @param  op2  [IN]  first operand
 * @return the new expression
 */
expr_t *expr_new_tertiar(const expr_t *cond, const expr_t *op1, const expr_t *op2);

#endif
