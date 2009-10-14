#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "expr.h"

#define EXPR_EVAL_ERROR_SIZE   512
static char expr_eval_error[EXPR_EVAL_ERROR_SIZE];

static int expr_eval_unary(unsigned char op, expr_t *op1, assignment_t *assignments, unsigned int nbassignments, int *v)
{
    int rc;
    int v1;
    
    assert( EXPR_IS_UNARY(op) );

    rc = expr_eval(op1, assignments, nbassignments, &v1);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    switch(op) {
    case EXPR_OP_UNARY_NOT:
        *v = !v1;
        break;
    }

    return EXPR_SUCCESS;
}

static int expr_eval_binary(unsigned char op, expr_t *op1, expr_t *op2, assignment_t *assignments, unsigned int nbassignments, int *v)
{
    int rc;
    int v1, v2;

    assert( EXPR_IS_BINARY(op) );

    rc = expr_eval(op1, assignments, nbassignments, &v1);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }
    rc = expr_eval(op2, assignments, nbassignments, &v2);
    if( EXPR_SUCCESS != rc ) {
        return rc;
    }

    switch(op) {
    case EXPR_OP_BINARY_MOD:
        *v = (v1 % v2);
        break;
    case EXPR_OP_BINARY_EQUAL:
        *v = (v1 == v2);
        break;
    case EXPR_OP_BINARY_PLUS:
        *v = (v1 + v2);
    }

    return EXPR_SUCCESS;
}

static int expr_eval_symbol(symbol_t *sym, assignment_t *assignments, unsigned int nbassignments, int *res)
{
    unsigned int i;

    for(i = 0; i < nbassignments; i++) {
        if( strcmp( sym->name, assignments[i].sym->name ) ) {
            continue;
        }
        *res = assignments[i].value;
        snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Success");
        return EXPR_SUCCESS;
    }

    snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Symbol not found in assignment: %s", sym->name);
    return EXPR_FAILURE_SYMBOL_NOT_FOUND;
}

int expr_eval(expr_t *expr, assignment_t *assignments, unsigned int nbassignments, int *res)
{
    if( EXPR_OP_SYMB == expr->op ) {
        return expr_eval_symbol(expr->var, assignments, nbassignments, res);
    } else if ( EXPR_OP_CONST_INT == expr->op ) {
        *res = expr->const_int;
        snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Success");
        return EXPR_SUCCESS;
    } else if ( EXPR_IS_UNARY(expr->op) ) {
        return expr_eval_unary(expr->op, expr->uop1, assignments, nbassignments, res);
    } else if ( EXPR_IS_BINARY(expr->op) ) {
        return expr_eval_binary(expr->op, expr->bop1, expr->bop2, assignments, nbassignments, res);
    } else {
        snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Unkown operand %d in expression", expr->op);
        return EXPR_FAILURE_UNKNOWN_OP;
    }
}

char *expr_error(void)
{
    return expr_eval_error;
}

void expr_dump(expr_t *e)
{
    printf("should dump the expression here\n");
}
