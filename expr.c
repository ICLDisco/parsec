#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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
        break;
    case EXPR_OP_BINARY_MINUS:
        *v = (v1 - v2);
        break;
    case EXPR_OP_BINARY_TIMES:
        *v = (v1 * v2);
        break;
    case EXPR_OP_BINARY_RANGE:
        snprintf(expr_eval_error, EXPR_EVAL_ERROR_SIZE, "Cannot evaluate range");
        return EXPR_FAILURE_CANNOT_EVALUATE_RANGE;
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

expr_t *expr_new_var(symbol_t *symb)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));
    r->op = EXPR_OP_SYMB;
    r->var = symb;
    return r;
}

expr_t *expr_new_int(int v)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));
    r->op = EXPR_OP_CONST_INT;
    r->const_int = v;
    return r;
}

expr_t *expr_new_binary(expr_t *op1, char op, expr_t *op2)
{
    expr_t *r = (expr_t*)calloc(1, sizeof(expr_t));

    r->bop1 = op1;
    r->bop2 = op2;
    
    switch( op ) {
    case '+':
        r->op = EXPR_OP_BINARY_PLUS;
        return r;
    case '-':
        r->op = EXPR_OP_BINARY_MINUS;
        return r;
    case '*':
        r->op = EXPR_OP_BINARY_TIMES;
        return r;
    case '%':
        r->op = EXPR_OP_BINARY_MOD;
        return r;
    case '=':
        r->op = EXPR_OP_BINARY_EQUAL;
        return r;
    case '.':
        r->op = EXPR_OP_BINARY_RANGE;
        return r;
    default:
        free(r);
        return NULL;
    }

    return r;
}

char *expr_error(void)
{
    return expr_eval_error;
}

static void expr_dump_unary(unsigned char op, const expr_t *op1)
{
    switch(op) {
    case EXPR_OP_UNARY_NOT:
        printf("!");
        break;
    }

    if( NULL == op1 ) {
        printf("NULL");
    } else {
        expr_dump(op1);
    }
}

static void expr_dump_binary(unsigned char op, const expr_t *op1, const expr_t *op2)
{
    expr_dump(op1);

    switch( op ) {
    case EXPR_OP_BINARY_PLUS:
        printf(" + ");
        break;
    case EXPR_OP_BINARY_MINUS:
        printf(" - ");
        break;
    case EXPR_OP_BINARY_TIMES:
        printf(" * ");
        break;
    case EXPR_OP_BINARY_MOD:
        printf(" %% ");
        break;
    case EXPR_OP_BINARY_EQUAL:
        printf(" == ");
        break;
    case EXPR_OP_BINARY_RANGE:
        printf(" .. ");
        break;
    }

    expr_dump(op2);
}

void expr_dump(const expr_t *e)
{
    if(NULL == e) {
        printf("NULL");
    }

    if( EXPR_OP_SYMB == e->op ) {
        printf("%s", e->var->name);
    } else if ( EXPR_OP_CONST_INT == e->op ) {
        printf("%d", e->const_int);
    } else if ( EXPR_IS_UNARY(e->op) ) {
        expr_dump_unary(e->op, e->uop1);
    } else if ( EXPR_IS_BINARY(e->op) ) {
        expr_dump_binary(e->op, e->bop1, e->bop2);
    } else {
        fprintf(stderr, "Unkown operand %d in expression", e->op);
    }    
}
