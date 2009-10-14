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
#define EXPR_OP_MAX_BINARY    22

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

#define EXPR_SUCCESS 0
#define EXPR_FAILURE_SYMBOL_NOT_FOUND   1
#define EXPR_FAILURE_UNKNOWN_OP         2

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
int expr_eval(expr_t *expr, assignment_t *assignments, unsigned int nbassignments, int *res);

/**
 * Gives some comments on the first error encountered during the last call to expr_eval
 *
 * @return static string to the comment. Undefined if expr_eval has not been called previously.
 */
char *expr_error(void);

#endif
